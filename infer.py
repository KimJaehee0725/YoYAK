import torch

from model.make_longformer import *
from transformers import PreTrainedTokenizerFast


def summarize_infer(
    text,
    model_ckpt="./longformer_kobart_trained_ckpt",
    tokenizer_ckpt="./longformer_kobart_initial_ckpt",
    max_length=1024,
    num_beams=5):
    
    source_max_len = 4096
    padding_idx = 3

    model = LongformerBartForConditionalGeneration.from_pretrained(model_ckpt)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_ckpt)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    text += "</s>"  # end token 추가
    input_ids = tokenizer.encode(text)
    input_ids += [padding_idx] * (source_max_len-len(input_ids))
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0) # input 형태에 따라 맞춰주어야 함

    output = model.generate(input_ids, eos_token_id=1, max_length=max_length, num_beams=num_beams)
    output = tokenizer.decode(output[0], skip_special_token=True)


    return output

if __name__ == "__main__":
    text = "안녕하세요"
    output = summarize_infer(text, model_ckpt='./longformer_kobart_initial_ckpt')
    print(output)

