# from kobart_transformers import get_kobart_tokenizer
from util_gsg import chainedLoader, getIndxs
from p_tqdm import p_umap
from tqdm import tqdm
import csv

# tokenizer = get_kobart_tokenizer()
# def tokenizeAndCorpus(corpus, tokenizer = tokenizer, getIndxs = getIndxs) :
#   mask_token_id = tokenizer("<mask>").input_ids
#   target_index = getIndxs(corpus)
#   target = [tokenizer(sentence).input_ids for indx, sentence in enumerate(corpus) if indx in target_index]
#   source = [tokenizer(sentence).input_ids if indx not in target_index else mask_token_id for indx, sentence in enumerate(corpus)]
#   return [source, target]

########### how to read files ###########
# import ast 
# import csv
# with open("G:/공유 드라이브/TobigsTextConf 141516/chained/chained_tokenized.csv", encoding = "utf-8", newline = "", mode = "r") as f :
#     reader = csv.reader(f)
#     corpus = list(reader)
# corpus_ = [[ast.literal_eval(line) for line in sentence] for sentence in corpus]







dataloader = chainedLoader()


def splitTargetSource(corpus, getIndxs = getIndxs) :
  mask_token = "<mask>"
  target_index = getIndxs(corpus)
  target = [sentence for indx, sentence in enumerate(corpus) if indx in target_index]
  source = [sentence if indx not in target_index else mask_token for indx, sentence in enumerate(corpus)]
  return [source, target]

def tokenizeByChunk (dataloader, chunk_size) :

  def processInMap(corpus):
    return splitTargetSource(corpus)

  chunk = []
  with tqdm(total = 3233442) as progress_bar :
    for num, line in enumerate(dataloader) :
      chunk.append(line)
      if (num + 1)%chunk_size == 0 :
        result = p_umap(processInMap, chunk, num_cpus = 7)
        with open("G:/공유 드라이브/TobigsTextConf 141516/chained/chained_SourceAndTarget.csv", encoding = "utf-8", newline = "", mode = "a") as f :
          writer = csv.writer(f)
          writer.writerows(result)
        progress_bar.update(chunk_size)
        chunk = []


    result = p_umap(processInMap, chunk)
    progress_bar.update(len(result))
    with open("G:/공유 드라이브/TobigsTextConf 141516/chained/chained_SourceAndTarget.csv", encoding = "utf-8", newline = "", mode = "a") as f :
      writer = csv.writer(f)
      writer.writerows(result)
    print(f"마지막 크기 : {len(result)} || 총 처리 크기 :{(num+1)*chunk_size + len(result)}")
    




def main():
  tokenizeByChunk(dataloader = dataloader, chunk_size = 10000)

if __name__ == '__main__':
  main()




