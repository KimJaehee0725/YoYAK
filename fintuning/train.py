from model_data_loader import *
from torch.utils.data import Dataloader, dataloader
model, tokenizer = load_YoYak()

dataset  = FineTuningDataset(tokenizer = tokenizer, data_path = " ")
data_loader = dataloader(dataset, batch_size = 4, collate_fn = collate_fn)
