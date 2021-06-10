import requests

class TfServingRequest():

    @staticmethod
    def char_matrix(tokens):
        chars = []
        max_len = max([len(token) for token in tokens])
        for token in tokens:
            token_chars = [ch for ch in token]
            for _ in range(max_len - len(token)):
                token_chars.append('')
            chars.append(token_chars)
        return chars

    @staticmethod
    def inputter(tokens, tokens_name = 'tokens', send_len = True, len_name = 'length', chars_name = 'chars'):
        data = {
            'signature_name': 'serving_default',
            'instances': [{
            }]
        }
        if chars_name:
            chars = TfServingRequest.char_matrix(tokens)
            data['instances'][0][chars_name] = chars

        if send_len:
            data['instances'][0][len_name] = len(tokens)
        
        data['instances'][0][tokens_name] = tokens
        # print('Sending', data, 'to tensorflow serving model')
        return data
    
    @staticmethod
    def request(request_url, tokens, tokens_name = 'tokens', output_token_name = 'tokens', send_len = True, len_name = 'length', chars_name = 'chars'):
        data = TfServingRequest.inputter(tokens, tokens_name, send_len, len_name, chars_name)

        serving_output = requests.post(request_url, json=data)
        status_code = serving_output.status_code

        if status_code == 200:
            output = TfServingRequest.outputter(serving_output.json(), output_token_name)
        else:
            output = serving_output.text

        return (output, status_code)
    
    @staticmethod
    def outputter(serving_output, output_token_name):
        output = serving_output['predictions'][0][output_token_name]
        return output
