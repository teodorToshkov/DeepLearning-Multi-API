import re

class Tokenizer(object):

    def __init__(self):
        pass
    
    def tokenize(self, sentence):
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)
        return self._tokenize(sentence)
    
    def _tokenize(self, sentence):
        tokens = re.findall(r'[а-яА-Яa-zA-Z0-9@]+|\.\.\.|[\:\;\!\?\.\"\,\'\(\)\[\]\{\}]', sentence)
        return tokens
    
    def detokenize(self, sentence):
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)
        return self._detokenize(sentence)
    
    def _detokenize(self, sentence):
        sentence = re.sub(r' ?<\/s>.*', '', sentence)
        sentence = sentence.replace(' <blank>', '')
        sentence = re.sub(r'\s(?=[\.\,\!\?\[\]\{\}\(\)])', '', sentence)
        return sentence
