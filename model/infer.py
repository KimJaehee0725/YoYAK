from make_longformer import *
from transformers import PreTrainedTokenizerFast

def load_model():
    model = LongformerBartForConditionalGeneration.from_pretrained('./longformer_kobart_trained_ckpt')
    return model

model = load_model()
tokenizer = PreTrainedTokenizerFast.from_pretrained('./longformer_kobart_initial_ckpt')

'''
Inference code
'''

source_max_len = 4096
padding_idx = 3

text = ''
input_ids = tokenizer.encode(text)
input_ids += [padding_idx] * (source_max_len-len(input_ids))
input_ids = torch.tensor(input_ids)
input_ids = input_ids.unsqueeze(0) # input 형태에 따라 맞춰주어야 함.
output = model.generate(input_ids, eos_token_id=1, max_length=1024, num_beams=5) # hyperparameter 조정 필요
output = tokenizer.decode(output[0], skip_special_tokens=True)
print(output)