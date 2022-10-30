from transformers import BertTokenizer
from transformers import BertModel
import torch
from torch.utils.data import  Dataset


class TextProcessor(Dataset):
    def __init__(self, max_length: int= 100):
       
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.model.eval()
        self.max_length = max_length
        

    def __call__(self, text):
        sentence = text
        encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        return description


if __name__ == '__main__':
    text_test = TextProcessor()
    var = text_test('dynamic range can not be overcome by simple aerodynamics')
    print(var)
    print(var.shape)