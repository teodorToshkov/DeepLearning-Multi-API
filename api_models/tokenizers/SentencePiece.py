import pyonmttok
import re

from api_models.tokenizers.Tokenizer import Tokenizer

class SentencePiece(Tokenizer):

    def __init__(self, sentencepiece_model):
        self._tokenizer = pyonmttok.Tokenizer('none', sp_model_path = sentencepiece_model)

    def _tokenize(self, sentence):
        tokens = self._tokenizer.tokenize(sentence)[0]
        return tokens
    
    def _detokenize(self, sentence):
        sentence = re.sub(r' ?<\/s>.*', '', sentence)
        sentence = sentence.replace(' <blank>', '')
        sentence = re.sub(r'\s(?=[\.\,\!\?\[\]\{\}\(\)])', '', sentence)
        sentence = ''.join(sentence.split()).replace('▁', ' ').strip()
        return sentence
