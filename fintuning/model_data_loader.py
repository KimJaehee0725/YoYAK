import torch

from model.make_longformer import *
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset

import pandas as pd

def load_YoYak(
    model_ckpt = "model/longformer_kobart_trained_ckpt",
    tokenizer_ckpt = "model/longformer_kobart_initial_ckpt",
    target_max_length = 1024,
    num_beams = 5,
    padding_idx = None):

    source_max_len = 4096
    if padding_idx == None : 
        padding_idx = 3
    
    model = LongformerBartForConditionalGeneration.from_pretrained(model_ckpt)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_ckpt)
    
    return model, tokenizer

class FineTuningDataset(Dataset):
    def __init__(self, tokenizer, data_path, pad_idx = None, encoder_max_len = 4095, decoder_max_len = 1022) : 
        super().__init()
        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.data = pd.read_csv(data_path)
        self.len = self.data.shape[0]

        if pad_idx == None:
            self.pad_idx = self.tokenizer.pad_token_id
        else :
            self.pad_idx = pad_idx
    
    def __getitem__(self, index) :
        instance = self.data.iloc[index]
        input_ids = self.tokenizer.encode(instance["source_text"])[:self.encoder_max_len]
        input_ids.append(self.tokenizer.eos_token_id)
        
        label_ids = self.tokenizer.encode(instance["summary_text"])[:self.decoder_max_len]
        label_ids.append(self.tokenizer.eos_token_id)
        
        decoder_input_ids = [self.tokenizer.eos_token_id]
        decoder_input_ids.append(label_ids)
    
        return {"input_ids" : input_ids,
        "decoder_input_ids" : decoder_input_ids,
        "label_ids" : label_ids}

def collate_fn(batch) :
    batch_size = batch.size()[0]
    pad_id = 3
    non_attention_value = 0
    not_cal_for_softmax = -100
    source_max_len  = 4096
    target_max_len = 1024
    
    encoder_tensor = torch.full(size = (batch_size, source_max_len), fill_value = pad_id, requires_grad = False)
    encoder_attention_tensor = torch.full(size = (batch_size, source_max_len), fill_value = non_attention_value, requires_grad = False)

    decoder_tensor = torch.full(size = (batch_size, target_max_len), fill_value = pad_id, requires_grad = False)
    decoder_attention_tensor = torch.full(size = (batch_size, target_max_len), fill_value = non_attention_value, requires_grad = False)

    label_tensor = torch.full(size = (batch_size, target_max_len), fill_value = not_cal_for_softmax, requires_grad = False)
    
    for num, data in enumerate(batch) :
        encoder_ids = data["input_ids"]
        encoder_len = encoder_ids.size()[0]
        encoder_tensor[num, :encoder_len] = encoder_ids
        encoder_attention_tensor[num, :encoder_len] = 1

        decoder_ids = data["decoder_input_ids"]
        decoder_len = decoder_ids.size()[0]
        decoder_tensor[num, :decoder_len] = decoder_ids
        decoder_attention_tensor[num, :decoder_len] = 1

        label_ids = data["label_ids"]
        label_tensor[num, :decoder_len - 1] = label_ids

        encoder_dict = {"input_ids" : encoder_tensor, "attention_mask" : encoder_attention_tensor}
        decoder_dict = {"input_ids" : decoder_tensor, "attention_mask" : decoder_attention_tensor}
        label_dict = {"input_ids" : label_tensor}

    return encoder_dict, decoder_dict, label_dict
    



