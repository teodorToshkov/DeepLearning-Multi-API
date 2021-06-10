"""

"""
import json
import os
import re
import subprocess
import sys

from api_models.ServableModel import ServableModel
from api_models.TfServingRequest import TfServingRequest
from api_models.tokenizers.BPE import *
from api_models.tokenizers.PyOnmtTok import PyOnmtTok
from api_models.tokenizers.SentencePiece import SentencePiece
from api_models.tokenizers.Tokenizer import Tokenizer


class OnmtSeq2Seq(ServableModel):
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

        self.tokenizer_type = 'simple'
        self.tokenizer = Tokenizer()
        if 'tokenizer' in config:
            self.tokenizer_type = config['tokenizer']

        if self.tokenizer_type == 'pyonmttok':
            self.tokenizer = PyOnmtTok()
            print(config)

        elif self.tokenizer_type == 'sentencepiece':
            self.tokenizer = SentencePiece(os.path.join(self.export_path, config['tokenizer-model']))
        
        if 'bpe-codes' in config:
            bpe_codes = os.path.join(self.export_path, config['bpe-codes'])
            bpe_vocab = os.path.join(self.export_path, config['bpe-vocab'])
            self.bpe = BPE(bpe_codes, vocab=bpe_vocab)
        else: self.bpe = None

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
        # command = ['tensorflow_model_server', '--port=' + str(self.rpc_port),
        #         '--rest_api_port=' + str(self.rest_port), '--model_name=' + self.name,
        #         '--model_base_path=' + '{base_dir}/{export_path}'.format(base_dir=os.getcwd(), export_path=self.export_path)]
        # self.model_process = subprocess.Popen(command)

    def process_request_string(self, request):
        """
        OK
        """
        data = request.data.decode('utf-8')

        if data == '':
            params = request.form
            sentences = json.loads(params['sent'])

        else:
            params = json.loads(data)
            sentences = params['sent']

        return sentences
    
    def detokenize(self, sentence_tokens):
        """
        OK
        """
        sentence = self.tokenizer.detokenize(sentence_tokens)
        if self.bpe:
            sentence = self.bpe.detokenize(sentence)
        return sentence

    def predict(self, sentence_tokens):
        serving_output, status_code = TfServingRequest.request(self.request_url, sentence_tokens, chars_name = None)

        if status_code != 200:
            print(status_code, serving_output[0], file=sys.stderr)
        return serving_output[0]

    def tokenize(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        if self.bpe:
            tokens = self.bpe.tokenize(tokens)
        return tokens
