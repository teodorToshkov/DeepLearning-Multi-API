"""

"""
import json
import os
import re
import subprocess
import sys
import time

from api_models.ServableModel import ServableModel

import requests
import json
import numpy as np
import scipy

from PIL import Image as pil_image

from io import BytesIO

class StyleTransfer(ServableModel):
    def __init__(self, config, name):
        """
        config in the form of:
            rpc-port: 9002
            rest-port: 8502
            rest-url: /v1/models/en-it:predict
            export-path: saved-models/en-it
            tokenizer: 'pyonmttok', 'sentnecepiece' (requires 'tokenizer-model'), 'simple'
            bpe-codes: bpe_codes.txt
            bpe-vocab: vocab.bpe.bg
            description: "English to Italian translation"
            # example: curl -X POST http://localhost:8881/en-it -d '{"sent": ["This is a simple test."]}' -H 'Content-Type: application/json'
            model-type: onmt-seq2seq
            timeout: 60 # in seconds
            active: true
        """
        super().__init__(config, name)

        self.rpc_port = config['rpc-port']
        self.rest_port = config['rest-port']
        self.rest_url = config['rest-url']
        self.export_path = config['export-path']
        self.request_url = 'http://127.0.0.1:{}{}'.format(self.rest_port, self.rest_url)

        if self.active:
            self.activate()


    def deactivate(self):
        """
        OK
        """
        super().deactivate()
        self.model_process.kill()

    def activate(self):
        """
        OK
        """
        super().activate()
        command = ['tensorflow_model_server', '--port=' + str(self.rpc_port),
                '--rest_api_port=' + str(self.rest_port), '--model_name=' + self.name,
                '--model_base_path=' + '{base_dir}/{export_path}'.format(base_dir=os.getcwd(), export_path=self.export_path)]
        print(command)
        self.model_process = subprocess.Popen(command)

    def process_request(self, request):
        if not self.active:
            self.activate()
            time.sleep(2)
        
        request = self.process_request_string(request)
        request = self.preproess(request)
        response = self.predict(request)
        response = self.postproess(response)

        return response

    def process_request_string(self, request):
        """
        OK
        """
        data = request.data.decode('utf-8')
        if data == '':
            params = request.form
            style = str(params['style'])
            content = str(params['content'])
            save_as = str(params['save_as'])

        else:
            params = json.loads(data)
            style = params['style']
            content = params['content']
            save_as = params['save_as']

        return [style, content, save_as]
    
    def postproess(self, response):
        """
        OK
        """
        [output, filename] = response
        out = [[[value * 255 for value in pixel] for pixel in row] for row in output]

        save_img('results/' + filename, out)

        print('finished', 'results/' + filename, file=sys.stderr)
        return 'results/' + filename

    def predict(self, request):
        [content, style, filename] = request
        print("Predict", filename, file=sys.stderr)
        payload = {"instances": [{'content_input': content.tolist(), 'style_input': style.tolist()}]}
        r = requests.post(self.request_url, json=payload)
        print(r.status_code, file=sys.stderr)
        if r.status_code != 200:
            print(r.text)
        output = json.loads(str(r.text))['predictions'][0]

        return [output, filename]

    def preproess(self, request):
        print(request, file=sys.stderr)
        [style, content, filename] = request

        style_image = img_to_array(load_img(style)) / 255.
        style_image = resize_to(style_image * 255, 512) / 255.
        image = img_to_array(load_img(content)) / 255.
        image = resize_to(image * 255, 512) / 255.

        return [image, style_image, filename]

def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if path.find('http') == 0\
        or path.find('ftp') == 0:
        response = requests.get(path)
        img = pil_image.open(BytesIO(response.content))
    else:
        img = pil_image.open('/app/saved-models/style-transfer-styles/' + path + '.jpg')

    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img

def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def resize_to(img, resize=512):
    '''Resize short side to target size and preserve aspect ratio'''
    height, width = img.shape[0], img.shape[1]
    if height < width:
        ratio = height / resize
        long_side = round(width / ratio)
        resize_shape = (resize, long_side, 3)
    else:
        ratio = width / resize
        long_side = round(height / ratio)
        resize_shape = (long_side, resize, 3)
    
    return scipy.misc.imresize(img, resize_shape, interp='bilinear')
