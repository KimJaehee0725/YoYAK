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
2. SOS, EOS 토큰 위치 확인해서 넣기 (현재 맨 앞에 CLS 토큰(2)이 들어가고 있음) 
3. iterabledataset multiprocessing 시 문제생기지 않는지 확인하기 o
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

def yield_token(corpus : list, tokenizer = get_kobart_tokenizer()) -> list:
    corpus = ["<s> " + line + " <\s>" for line in corpus]
    full_sentence = " ".join(corpus)
    return tokenizer(full_sentence, return_tensor = "pt")['input_ids']

def collat_batch(batch):
    source_max_len = 4096
    target_max_len = 512
    batch_size = len(batch)

    source_tensor = torch.zeros(batch_size, source_max_len, dtype = torch.int64)
    target_tensor = torch.zeros(batch_size, target_max_len, dtype = torch.int64)
    
    for num, (source, target) in enumerate(batch):
        source_preprocessed = torch.tensor(yield_token(source))
        source_len = len(source_preprocessed)
        if source_len > source_max_len :
            print("source 문장의 토큰 수가 4096을 넘습니다.")
        source_tensor[num, :source_len] = source_preprocessed

        target_preprocessed = torch.tensor(yield_token(target))
        target_len = len(target_preprocessed)
        if target_len > target_max_len :
            print("target 문장의 토큰 수가 512를 넘습니다.")
        target_tensor[num, :target_len] = target_preprocessed
    return source_tensor, target_tensor

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

