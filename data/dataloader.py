from transformers import AutoTokenizer
import torch
import csv
from torch.utils.data import DataLoader



def data_iter(path):
    with open(path, mode = "r", newline = "", encoding = "utf-8-sig") as f:
        reader = csv.reader(f)
        return list(reader)
        # for line in reader :
        #     yield list(line)

def yield_token(corpus : list, tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")) -> list:
    full_sentence = " ".join(corpus)
    return tokenizer(full_sentence)['input_ids']

def collat_batch(batch):
    text_list = []
    for text in batch:
        processed_text = torch.tensor(yield_token(text))
        text_list.append(processed_text)
    torch.cat(text_list)
    return text_list




data = data_iter('G:/공유 드라이브/TobigsTextConf 141516/cleaned/namuwiki_cleaned.csv')
data_loader = DataLoader(data, batch_size = 4, shuffle = True, collate_fn = collat_batch)
data_loader = iter(data_loader)
print(next(data_loader))

