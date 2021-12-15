from kobart_transformers import get_kobart_tokenizer
import torch
import csv
from torch.utils.data import DataLoader, IterableDataset
import ast 
import csv
from torch.nn.utils.rnn import pad_sequence
"""
todo 
1. pad_sequence의 길이를 ToBigBird의 최대길이(4096)으로 맞추기 o
2. iterabledataset multiprocessing 시 문제생기지 않는지 확인하기 o
3. SOS, EOS 토큰 위치 확인해서 넣기 (완료된듯...? 확신이 없음)
    => source의 경우 문장마다 <s>sentence</s> 로 넣으면 되고, target의 경우엔 </s>sentence</s>로 넣으면 됨. 
    이유는 https://stackoverflow.com/questions/64904840/why-we-need-a-decoder-start-token-id-during-generation-in-huggingface-bart 참고
    다만 target에서 매 sentence마다 </s>로 시작하는지는 확신이 없음. 페가수스 관련 코드 찾아봐야 할듯. 
4. mask 토큰이 기존에는 token 단위인데, 우린 sentence 단위로 masking하고 있어서 다를 수 있음. 이를 해결하기 위해 새로운 토큰으로 mask_new 토큰을 만들어야 하는지 고민해보기. 
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
    corpus = ["<s>" + line + "</s>" for line in corpus]
    full_sentence = "".join(corpus)
    return tokenizer(full_sentence)['input_ids']

def yield_target(corpus : list, tokenizer = get_kobart_tokenizer()) -> list:
    corpus = ["</s>" + line + "</s>" for line in corpus]
    full_sentence = "".join(corpus)
    return tokenizer(full_sentence)['input_ids']

def collat_batch(batch):
    pad_id = 3
    non_attention_value = 0
    source_max_len = 4096
    target_max_len = 512
    batch_size = len(batch)

    source_token_ids = torch.full(size = (batch_size, source_max_len), fill_value = pad_id, dtype = torch.int, requires_grad = False)
    source_attention_masks = torch.full(size = (batch_size, source_max_len), fill_value = non_attention_value, dtype = torch.int, requires_grad = False)

    target_token_ids = torch.full(size = (batch_size, target_max_len), fill_value = pad_id, dtype = torch.int, requires_grad = False)
    target_attention_masks = torch.full(size = (batch_size, target_max_len), fill_value = non_attention_value, dtype = torch.int, requires_grad = False)
    
    
    for num, (source, target) in enumerate(batch):
        source_preprocessed = torch.tensor(yield_source(source), requires_grad = False)
        source_len = len(source_preprocessed)
        if source_len > source_max_len :
            print("source 문장의 토큰 수가 4096을 넘습니다.")
        source_token_ids[num, :source_len] = source_preprocessed[:source_len]
        source_attention_masks[num, :source_len] = 1

        target_preprocessed = torch.tensor(yield_target(target), requires_grad = False)
        target_len = len(target_preprocessed)
        if target_len > target_max_len :
            print("target 문장의 토큰 수가 512를 넘습니다.")
        target_token_ids[num, :target_len] = target_preprocessed[:target_len]
        target_attention_masks[num, :target_len] = 1

    source_dict = {"token_ids" : source_token_ids, "attention_mask" : source_attention_masks}
    target_dict = {"token_ids" : target_token_ids, "attention_mask" : target_attention_masks}
    return source_dict, target_dict

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset.data)//worker_info.num_workers

    dataset.data = dataset.data[worker_id*split_size : (worker_id + 1)*split_size]


"""
how to use
-------------------------------------------
data_iterator = iterableDataset()
data_loader = DataLoader(data_iterator, batch_size = 4, collate_fn = collat_batch, num_workers = 2, worker_init_fn = worker_init_fn)
data_loader = iter(data_loader)
print(next(data_loader))
"""

from torch.utils.data import DataLoader
data = iterableDataset()
data_loader = DataLoader(data, batch_size = 4, collate_fn = collat_batch)
data_loader = iter(data_loader)
x, y = next(data_loader)
print(x)
print(y)