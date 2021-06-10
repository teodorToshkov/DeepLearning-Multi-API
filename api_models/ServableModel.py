from abc import ABC, abstractmethod
import time

class ServableModel:
    """

    """
    def __init__(self, config, name):
        self.config = config
        self.description = config['description']
        self.timeout = config['timeout']
        self.active = config['active']
        self.name = name
    
    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def process_request(self, request):
        if not self.active:
            self.activate()
            time.sleep(2)
        
        sentences = self.process_request_string(request)
        tokenized_sentences = [self.tokenize(sentence) for sentence in sentences]
        response = [self.predict(sentence_tokens) for sentence_tokens in tokenized_sentences]
        detokenized_response = [self.detokenize(sentence) for sentence in response]

        return detokenized_response
    
    @abstractmethod
    def process_request_string(self, request):
        pass

    @abstractmethod
    def predict(self, sentence_tokens):
        pass
    
    def tokenize(self, sentence):
        return sentence
    
    def detokenize(self, sentence_tokens):
        return sentence_tokens

class MyRequest():
    def __init__(self, data):
        self.data = data.encode()
