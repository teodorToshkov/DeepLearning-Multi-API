from api_models.OnmtSeq2seq import OnmtSeq2Seq
from api_models.TfServingRequest import TfServingRequest

import sys
import time

class SeqTagger(OnmtSeq2Seq):

    def detokenize(self, sentence_tokens, input_tokens):
        output = []
        for i in range(len(sentence_tokens)):
            output.append({
                'word': input_tokens[i],
                'tag': sentence_tokens[i]
            })
        return output

    def predict(self, sentence_tokens):
        serving_output, status_code = TfServingRequest.request(self.request_url, sentence_tokens, chars_name = 'chars', output_token_name = 'tags')

        if status_code != 200:
            print(status_code, serving_output, file=sys.stderr)
        return serving_output

    def process_request(self, request):
        if not self.active:
            self.activate()
            time.sleep(2)
        
        sentences = self.process_request_string(request)
        tokenized_sentences = [self.tokenize(sentence) for sentence in sentences]
        response = [self.predict(sentence_tokens) for sentence_tokens in tokenized_sentences]
        detokenized_response = [self.detokenize(sentence, tokenized_sentences[i]) for i, sentence in enumerate(response)]

        return detokenized_response
    