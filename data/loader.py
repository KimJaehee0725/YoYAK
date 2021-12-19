from kobart_transformers import get_kobart_tokenizer
import torch
import csv
from torch.utils.data import DataLoader, IterableDataset
import ast 
import csv
from torch.nn.utils.rnn import pad_sequence
"""""

1. pad_sequence의 길이를 ToBigBird의 최대길이(4096)으로 맞추기 o
2. iterabledataset multiprocessing 시 문제생기지 않는지 확인하기 o

3̶.̶ S̶O̶S̶,̶ E̶O̶S̶ 토̶큰̶ 위̶치̶ 확̶인̶해̶서̶ 넣̶기̶ (̶완̶료̶된̶듯̶.̶.̶.̶?̶ 확̶신̶이̶ 없̶음̶)̶
    =̶>̶ s̶o̶u̶r̶c̶e̶의̶ 경̶우̶ 문̶장̶마̶다̶ <̶s̶>̶s̶e̶n̶t̶e̶n̶c̶e̶<̶/̶s̶>̶ 로̶ 넣̶으̶면̶ 되̶고̶,̶ t̶a̶r̶g̶e̶t̶의̶ 경̶우̶엔̶ <̶/̶s̶>̶s̶e̶n̶t̶e̶n̶c̶e̶<̶/̶s̶>̶로̶ 넣̶으̶면̶ 됨̶.̶ 
    이̶유̶는̶ h̶t̶t̶p̶s̶:̶/̶/̶s̶t̶a̶c̶k̶o̶v̶e̶r̶f̶l̶o̶w̶.̶c̶o̶m̶/̶q̶u̶e̶s̶t̶i̶o̶n̶s̶/̶6̶4̶9̶0̶4̶8̶4̶0̶/̶w̶h̶y̶-̶w̶e̶-̶n̶e̶e̶d̶-̶a̶-̶d̶e̶c̶o̶d̶e̶r̶-̶s̶t̶a̶r̶t̶-̶t̶o̶k̶e̶n̶-̶i̶d̶-̶d̶u̶r̶i̶n̶g̶-̶g̶e̶n̶e̶r̶a̶t̶i̶o̶n̶-̶i̶n̶-̶h̶u̶g̶g̶i̶n̶g̶f̶a̶c̶e̶-̶b̶a̶r̶t̶ 참̶고̶
    다̶만̶ B̶a̶r̶t̶는̶ e̶n̶c̶o̶d̶e̶r̶와̶ d̶e̶c̶o̶d̶e̶r̶에̶ 한̶ 문̶장̶만̶ 입̶력̶하̶기̶ 때̶문̶에̶,̶ t̶a̶r̶g̶e̶t̶에̶서̶ 매̶ s̶e̶n̶t̶e̶n̶c̶e̶마̶다̶ <̶/̶s̶>̶로̶ 시̶작̶하̶는̶지̶는̶ 확̶신̶이̶ 없̶음̶.̶
3̶-̶1̶.̶ p̶e̶g̶a̶s̶u̶s̶ t̶o̶k̶e̶n̶i̶z̶e̶r̶ 결̶과̶를̶ 한̶번̶ 보̶기̶.̶
    관̶련̶ 링̶크̶ :̶ h̶t̶t̶p̶s̶:̶/̶/̶g̶i̶t̶h̶u̶b̶.̶c̶o̶m̶/̶h̶u̶g̶g̶i̶n̶g̶f̶a̶c̶e̶/̶t̶r̶a̶n̶s̶f̶o̶r̶m̶e̶r̶s̶/̶i̶s̶s̶u̶e̶s̶/̶1̶1̶5̶4̶1̶
    o̶u̶t̶p̶u̶t̶을̶ 살̶펴̶보̶니̶ s̶e̶n̶t̶e̶c̶e̶1̶.̶<̶n̶>̶s̶e̶n̶t̶e̶n̶c̶e̶2̶.̶<̶n̶>̶ 이̶런̶ 식̶으̶로̶ 나̶옴̶.̶ 근̶데̶ 아̶마̶ \̶n̶ 대̶신̶에̶ 저̶걸̶ 쓴̶듯̶
    a̶d̶d̶i̶t̶i̶o̶n̶a̶l̶ u̶n̶k̶ 토̶큰̶들̶이̶ 있̶으̶니̶까̶ 이̶걸̶ 이̶용̶해̶서̶ 우̶리̶ 나̶름̶대̶로̶ 꾸̶려̶봐̶야̶ 할̶ 듯̶.̶

4. mask 토큰이 기존에는 token 단위인데, 우린 sentence 단위로 masking하고 있어서 다를 수 있음. 이를 해결하기 위해 새로운 토큰으로 mask_new 토큰을 만들어야 하는지 고민해보기 o
-> unused0 사용

5. special token 사용 & 입력값 형태
-> encoder input : <sos>sentence<eos><pad><pad>...
-> encoder attention mask : 1 1 1 1 1 1 0 0 0...
-> decoder input : <eos>sentence<eos><pad><pad><pad>
-> decoder attention mask : 1 1 1 1 1 1 0 0 0 0
-> label : sentence-100-100-100
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
    corpus = ["<s>" + line.replace("<mask>", "<unused0>") + "</s>" for line in corpus]
    full_sentence = "".join(corpus)
    return tokenizer(full_sentence)['input_ids']

def yield_target(corpus : list, tokenizer = get_kobart_tokenizer()) -> list:
    corpus = ["</s>" + line.replace("<mask>", "<unused0>") + "</s>" for line in corpus]
    full_sentence = "".join(corpus)
    return tokenizer(full_sentence)['input_ids']

def collat_batch(batch):
    pad_id = 3
    non_attention_value = 0
    not_cal_for_softmax = -100
    source_max_len = 4096
    target_max_len = 1024
    batch_size = len(batch)

    source_token_ids = torch.full(size = (batch_size, source_max_len), fill_value = pad_id, dtype = torch.int, requires_grad = False)
    source_attention_masks = torch.full(size = (batch_size, source_max_len), fill_value = non_attention_value, dtype = torch.int, requires_grad = False)

    target_token_ids = torch.full(size = (batch_size, target_max_len), fill_value = pad_id, dtype = torch.int, requires_grad = False)
    target_attention_masks = torch.full(size = (batch_size, target_max_len), fill_value = non_attention_value, dtype = torch.int, requires_grad = False)
    
    label_token_ids = torch.full((batch_size, target_max_len), fill_value = not_cal_for_softmax, dtype = torch.int, requires_grad = False)
    
    for num, (source, target) in enumerate(batch):
        source_preprocessed = torch.tensor(yield_source(source), requires_grad = False)
        source_len = len(source_preprocessed)
        if source_len > source_max_len :
            print(f"source 문장의 토큰 수가 {source_max_len}을 넘습니다.")
        source_token_ids[num, :source_len] = source_preprocessed[:source_len]
        source_attention_masks[num, :source_len] = 1

        target_preprocessed = torch.tensor(yield_target(target), requires_grad = False)
        target_len = len(target_preprocessed)
        if target_len > target_max_len :
            print(f"target 문장의 토큰 수가 {target_max_len}를 넘습니다.")
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


"""
how to use
-------------------------------------------
data_iterator = iterableDataset()
data_loader = DataLoader(data_iterator, batch_size = 4, collate_fn = collat_batch, num_workers = 2, worker_init_fn = worker_init_fn)
data_loader = iter(data_loader)
print(next(data_loader))
"""
