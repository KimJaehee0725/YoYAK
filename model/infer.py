import torch
import torch.nn.functional as F

from make_longformer import *
from transformers import PreTrainedTokenizerFast


def summarize_infer(
    text,
    model_ckpt="./longformer_kobart_trained_ckpt",
    tokenizer_ckpt="./longformer_kobart_initial_ckpt",
    max_length=1024,
    num_beams=5):
    
    max_input_seq_len = 4096
    pad_token_id = 3

    model = LongformerBartForConditionalGeneration.from_pretrained(model_ckpt)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_ckpt)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    text += "</s>"  # end token 추가
    input_ids = torch.tensor(tokenizer.encode(text), device=device).unsqueeze(0)
    padding_len = max_input_seq_len - len(input_ids[0])
    input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)

    output = model.generate(input_ids, eos_token_id=1, max_length=max_length, num_beams=num_beams)
    output = tokenizer.decode(output[0], skip_special_token=True)

    return output

if __name__ == "__main__":
    text = "안녕하세요"
    output = summarize_infer(text, model_ckpt='./longformer_kobart_initial_ckpt')
    print(output)