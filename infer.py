import torch

from model.make_longformer import *
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
def load_model(model_ckpt, tokenizer_ckpt):
    if "YoYak" in model_ckpt:
        model = LongformerBartForConditionalGeneration.from_pretrained(model_ckpt)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_ckpt)
    else :
        model = BartForConditionalGeneration.from_pretrained(model_ckpt)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_ckpt)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return model, tokenizer, device

def summarize_batch_infer(
    text_list,
    model, 
    tokenizer,
    device,
    target_max_length=1024,
    source_max_len = 4096,
    num_beams=5) :
    padding_idx = 3

    text_list = [text + "</s>" for text in text_list]  # end token 추가
    input_ids = tokenizer(text_list, return_tensors = "pt", truncation = True, padding = True, max_length = 555).input_ids

    if input_ids.size()[1] < target_max_length:
        pad_tensor = torch.full(size = (input_ids.size()[0], source_max_len - input_ids.size()[1]), fill_value = padding_idx)
        input_ids = torch.cat((input_ids, pad_tensor), dim = 1)
    
    else:
        input_ids = input_ids[:, :target_max_length] # input 형태에 따라 맞춰주어야 함

    input_ids = input_ids.to(device)
    output = model.generate(input_ids, eos_token_id=1, max_length=target_max_length, num_beams=num_beams)
    output = tokenizer.batch_decode(output, skip_special_tokens=True)
    output = [text + "\n" for text in output]
    return output   


def summarize_infer(
    text,
    model, 
    tokenizer,
    device,
    max_length=1024,
    num_beams=5):
    
    source_max_len = 4096
    padding_idx = 3

    text += "</s>"  # end token 추가
    input_ids = tokenizer.encode(text)
    input_ids += [padding_idx] * (source_max_len-len(input_ids))
    input_ids = torch.tensor(input_ids, device=device)
    input_ids = input_ids.unsqueeze(0) # input 형태에 따라 맞춰주어야 함

    output = model.generate(input_ids, eos_token_id=1, max_length=max_length, num_beams=num_beams)
    output = tokenizer.decode(output[0], skip_special_token=True)

    return output

if __name__ == "__main__":
    text = "안녕하세요"
    model, tokenizer, device = load_model(model_ckpt='model/longformer_kobart_initial_ckpt', tokenizer_ckpt = "model/longformer_kobart_initial_ckpt")
    output = summarize_infer(text, model = model, tokenizer = tokenizer, device = device)
    print(output)

