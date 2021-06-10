import yaml
import re
import requests
import json
import subprocess
import os

from flask import Flask, request
from flask_cors import CORS

from api_models.FillInTheBlanks import FillInTheBlanks
from api_models.ServableModel import ServableModel
from api_models.OnmtSeq2seq import OnmtSeq2Seq
from api_models.SequenceTagger import SeqTagger
from api_models.StyleTransfer import *

app = Flask(__name__)
cors = CORS(app)

config = yaml.load('\n'.join(open('config.yml', encoding='utf-8').readlines()))
models_config = config['models']
models = {}
print(config)

from flask import send_from_directory

# ======== Style Transfer helpers =========
@app.route('/image/<name>', methods=['GET'])
def _get_img(name):
    print("Got request for image", name, file=sys.stderr)
    return send_from_directory('./results/', name)

@app.route('/styles', methods=['GET'])
def list_styles():
    return '; '.join(['.'.join(f.split('.')[0:-1]) for f in os.listdir('/app/saved-models/style-transfer-styles/') if f.split('.')[-1] == 'jpg'])
# =========================================

@app.route('/<model_name>', methods=['POST'])
def predict(model_name):
    print(model_name)
    response = models[model_name].process_request(request)
    return json.dumps(response, ensure_ascii=False)

# curl -X POST 127.0.0.1:8881/bg-pos -d '{"sent": ["Това е тест"]}' -H 'Content-Type: application/json'
# response = request_service('http://127.0.0.1:8500/v1/models/bg-pos:predict', 'sequence-tagger', 'Това е тест')
# print(u''.join(json.dumps(response, ensure_ascii=False)))

def init_model(model_config, model_name):
    if model_config['model-type'] == 'onmt-seq2seq':
        models[model_name] = OnmtSeq2Seq(model_config, model_name)
    elif model_config['model-type'] == 'sequence-tagger':
        models[model_name] = SeqTagger(model_config, model_name)
    elif model_config['model-type'] == 'fill-in-the-blanks':
        pos_tagger_name = model_config['pos-tagger']
        models[model_name] = FillInTheBlanks(model_config, model_name, models[pos_tagger_name])
    elif model_config['model-type'] == 'style-transfer':
        models[model_name] = StyleTransfer(model_config, model_name)

if __name__ == '__main__':
    print('Starting the models...')
    init_model(models_config['en-pos'], 'en-pos')
    for model_name in models_config:
        model_config = models_config[model_name]
        init_model(model_config, model_name)
    print('Running the API...')
    app.run(host=config['host'], debug=True, port=config['port'], threaded=True)
