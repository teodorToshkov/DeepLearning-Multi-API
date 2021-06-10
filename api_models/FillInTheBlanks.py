
from api_models.ServableModel import ServableModel, MyRequest

from gensim.models import KeyedVectors
import json

class FillInTheBlanks(ServableModel):

    def __init__(self, config, name, pos_tagger):
        """
        fill-in-the-blanks:
            pos-tagger: en-pos
            embeddings-file: saved-models/fill-in-the-blanks/glove-word-embeddings/multilingual.gensim
            description: "Creates a fill-in-the-blanks exercise"
            # example: curl -X POST http://localhost:8881/fill-in-the-blanks -d '{"lang": "en", "sent": ["This is a test."]}' -H 'Content-Type: application/json'
            model-type: fill-in-the-blanks
            timeout: 60 # in seconds
            active: true
        """
        self.pos_tagger = pos_tagger    
        self.embeddings_file = config['embeddings-file']

        super().__init__(config, name)

        if self.active:
            self.activate()

    def predict(self, sentence, lang='en', tanslate_lang='en'):

        if len(sentence) == 0:
            pass
        for word in sentence:
            if (word['tag'] == 'nn' or word['tag'] == 'np')\
            and lang + ':' + word['word'].lower() in self.embeddings.wv.vocab:
                word['synonyms'] =\
                    [
                        (w[0][3:], w[1]) for w in self.embeddings.similar_by_word(lang + ":" + word['word'].lower(), 10000)
                        if lang + ':' in w[0]
                    ][:10]
                word['translation'] =\
                    [
                        (w[0][3:], w[1]) for w in self.embeddings.similar_by_word(lang + ":" + word['word'].lower(), 1000)
                        if tanslate_lang + ':' in w[0]
                    ][0]

        return sentence

    def process_request(self, request):
        if not self.active:
            self.activate()
        
        sentences, lang = self.process_request_string(request)
        tokenized_sentences = [self.tokenize(sentence) for sentence in sentences]
        response = [self.predict(sentence_tokens, tanslate_lang=lang) for sentence_tokens in tokenized_sentences]
        detokenized_response = [self.detokenize(sentence) for sentence in response]

        return detokenized_response
    

    def process_request_string(self, request):
        """
        OK
        """
        data = request.data.decode('utf-8')

        if data == '':
            params = request.form
            sentences = json.loads(params['sent'])
            lang = json.loads(params['lang'])

        else:
            params = json.loads(data)
            sentences = params['sent']
            lang = params['lang']

        return (sentences, lang)
        
    def activate(self):
        super().activate()
        self.embeddings = KeyedVectors.load(self.embeddings_file)

    def deactivate(self):
        super().deactivate()
        del self.embeddings

    def tokenize(self, sentence):
        request = MyRequest(json.dumps({
            'sent': [sentence]
        }))
        tagged_sentence = self.pos_tagger.process_request(request)[0]
        return tagged_sentence
    
    def detokenize(self, sentence_tokens):
        return sentence_tokens
