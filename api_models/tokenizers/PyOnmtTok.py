import pyonmttok
import re

from api_models.tokenizers.Tokenizer import Tokenizer

class PyOnmtTok(Tokenizer):

    def __init__(self):
        self._tokenizer = pyonmttok.Tokenizer(
                            'aggressive',
                            joiner=('￭'),
                            joiner_annotate=True,
                            spacer_annotate=True,
                            segment_case=True,
                            segment_numbers=True)

    def _tokenize(self, sentence):
        tokens = self._tokenizer.tokenize(sentence)[0]
        return tokens
    
    def _detokenize(self, sentence):
        sentence = re.sub(r'\s*￭\s*', '', sentence)
        sentence = re.sub(r' ?<\/s>.*', '', sentence)
        sentence = sentence.replace(' <blank>', '')
        sentence = re.sub(r'\s(?=[\.\,\!\?\[\]\{\}\(\)])', '', sentence)
        return sentence
