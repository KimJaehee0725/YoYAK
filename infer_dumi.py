import pandas as pd
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import csv
import gc
from tqdm.notebook import tqdm
from infer import load_model, summarize_batch_infer

test_file_path = "/content/drive/Shareddrives/TobigsTextConf 141516/finetuning_data/test_over_dumi_.tsv"
test_data = pd.read_csv(test_file_path, index_col = 0)[:17]

model_ckpt="/content/drive/Shareddrives/TobigsTextConf 141516/finetuning_data/kobart_finetuned_checkpoint" # "/content/drive/Shareddrives/TobigsTextConf 141516/finetuning_data/YoYak_final_finetuned_ckpt
tokenizer_ckpt='gogamza/kobart-base-v2' # /content/drive/Shareddrives/TobigsTextConf 141516/longformer_checkpoint

model, tokenizer, device = load_model(model_ckpt, tokenizer_ckpt)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    batch_size = 3
    with open(r"/content/drive/Shareddrives/TobigsTextConf 141516/inference_result/dumi/test_under_infer.tsv", "a", encoding="utf-8") as f:
        tw = csv.writer(f, delimiter="\t")
        tw.writerow(['source', 'label', 'summary', 'dumi'])
        for i in tqdm(range((len(test_data)//batch_size)+1)):
            text = test_data.iloc[i*batch_size:(i+1)*batch_size,0]
            label = test_data.iloc[i*batch_size:(i+1)*batch_size,1]
            dumi = test_data.iloc[i*batch_size:(i+1)*batch_size,2]
            output = summarize_batch_infer(text, model, tokenizer, device, target_max_length = 512, source_max_len = 512) #1024, 4096
            for j in zip(text, label, output, dumi):
                tw.writerow(list(j))
            gc.collect()
            torch.cuda.empty_cache()