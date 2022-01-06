
import csv
import ast 
import torch
from torch.utils.data import IterableDataset
from kobart_transformers import get_kobart_tokenizer

"""
how to use
-------------------------------------------
data_iterator = iterableDataset()
data_loader = DataLoader(data_iterator, batch_size = 4, collate_fn = collat_batch, drop_last = True)
data_loader = iter(data_loader)
print(next(data_loader))

5. special token 사용 & 입력값 형태
-> encoder input : sentence<eos><pad><pad>...
-> encoder attention mask : 1 1 1 1 1 1 0 0 0...
-> decoder input : <eos>sentence<eos><pad><pad><pad>
-> decoder attention mask : 1 1 1 1 1 1 0 0 0 0
-> label : sentence<eos>100100100
"""

class iterableDataset(IterableDataset):
    def __init__(self, path = "G:/공유 드라이브/TobigsTextConf 141516/chained/chained_SourceAndTarget.csv"):
        super().__init__()
        self.path = path
        
    def __iter__(self) :
        with open(self.path, encoding = "utf-8", newline = "", mode = "r") as f :
            reader = csv.reader(f)
            for corpus in reader:
                corpus = list(corpus)
                source, target = (ast.literal_eval(line) for line in corpus)
                yield source, target

def yield_source(corpus : list, tokenizer = get_kobart_tokenizer()) -> list:
    corpus = [line.replace("<mask>", "<unused0>") + "</s>" for line in corpus]
    full_sentence = "".join(corpus)
    return tokenizer(full_sentence)['input_ids']

def yield_target(corpus : list, tokenizer = get_kobart_tokenizer()) -> list:
    corpus = ["</s>" + line.replace("<mask>", "<unused0>") + "</s>" for line in corpus]
    full_sentence = "".join(corpus)
    return tokenizer(full_sentence)['input_ids']

def collat_batch(batch):
    pad_id = 3
    sos_id = 0
    eos_id = 1
    non_attention_value = 0
    not_cal_for_softmax = -100
    source_max_len = 4096
    target_max_len = 1024
    batch_size = len(batch)

    source_token_ids = torch.full(size = (batch_size, source_max_len), fill_value = pad_id, requires_grad = False)
    source_attention_masks = torch.full(size = (batch_size, source_max_len), fill_value = non_attention_value, requires_grad = False)

    target_token_ids = torch.full(size = (batch_size, target_max_len), fill_value = pad_id,  requires_grad = False)
    target_attention_masks = torch.full(size = (batch_size, target_max_len), fill_value = non_attention_value,  requires_grad = False)
    
    label_token_ids = torch.full(size = (batch_size, target_max_len), fill_value = not_cal_for_softmax,  requires_grad = False)
    
    for num, (source, target) in enumerate(batch):
        source_preprocessed = torch.tensor(yield_source(source), requires_grad = False)
        source_len = len(source_preprocessed)
        if source_len > source_max_len :

            # print(f"source 문장의 토큰 수가 {source_max_len}을 넘습니다.")
            source_preprocessed = source_preprocessed[:source_max_len]
            source_preprocessed[source_max_len-1] = eos_id
            source_len = len(source_preprocessed)
        
        if source_len == 0:
            source_preprocessed = torch.tensor([sos_id, eos_id])
            source_len = len(source_preprocessed)
            
        source_token_ids[num, :source_len] = source_preprocessed[:source_len]
        source_attention_masks[num, :source_len] = 1

        target_preprocessed = torch.tensor(yield_target(target), requires_grad = False)
        target_len = len(target_preprocessed)

        if target_len > target_max_len :
            # print(f"target 문장의 토큰 수가 {target_max_len}를 넘습니다.")
            target_preprocessed = target_preprocessed[:target_max_len]
            target_preprocessed[target_max_len-1] = eos_id
            target_len = len(target_preprocessed)
        
        if target_len == 0:
            target_preprocessed = torch.tensor([sos_id, eos_id])
            target_len = len(target_preprocessed)

        target_token_ids[num, :target_len] = target_preprocessed[:target_len]
        target_attention_masks[num, :target_len] = 1 

        label = target_preprocessed[1:]
        label_token_ids[num, :target_len-1] = label
    
    source_dict = {"token_ids" : source_token_ids, "attention_mask" : source_attention_masks}
    target_dict = {"token_ids" : target_token_ids, "attention_mask" : target_attention_masks}
    label_dict = {"token_ids" : label_token_ids}
    return source_dict, target_dict, label_dict

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset.data)//worker_info.num_workers

    dataset.data = dataset.data[worker_id*split_size : (worker_id + 1)*split_size]



