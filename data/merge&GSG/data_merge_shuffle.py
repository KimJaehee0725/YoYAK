import csv
import random
import time
from itertools import zip_longest
from typing import Iterable
from tqdm import tqdm
###############
# todo : 
# 1. chunk로 뭉텅이로 읽고 저장하도록 코드 짜기 o
# 1-1. len_leftover가 저장되지 않는듯 하다. 이를 수정하자. x
# 2. chained를 iterator로 만들기 x 
# 3. 마지막 남은 애들 저장하는 코드 짜기(buff랑 dataloader에 있는 거 모두) x
###############


def makeMaskForDataLoader(len_list : list) -> list:
    result = []
    accum = 0
    for len_ in len_list:
        accum += len_
        result.append(accum)
    return result

def selectDataloader(ratio_list : list):
    seed = random.random()
    for idx, ratio in enumerate(ratio_list):
        if seed <= ratio:
            return idx

class IterableDatasetCustom:
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = data_path.split("/")[-1]
        self.len = 0

        print(f"Started Counting {self.dataset_name}")

        with open(data_path, mode = "r", newline="", encoding = "utf-8-sig") as dataset:
            reader = csv.reader(dataset)
            for num, line in enumerate(reader):
                self.len += 1

                if num%10000 == 0:
                    print(f"{num} Corpus was read", end = "\r")



        print(f"Finished Counting {self.dataset_name}")
        print(f"The Length of {self.dataset_name} dataset is {self.len}")
        print("-"*100)

    
    def __iter__(self):
        with open(self.data_path, mode = "r", newline="", encoding = "utf-8-sig") as dataset:
            reader = csv.reader(dataset)
            for line in reader:
                yield list(line)
    
    def __len__(self):
        return self.len

class chainMultiIterableDataset :
  def __init__(self, root_path : str, dataset_name : list, buff_size = 100, chunk_size = 10):
    super().__init__()
    """
    buff : shuffling을 위해 클래스 인스터스 내에 임시로 저장하는 코퍼스
    buff_size : shuffling을 위해 이용할 buff의 크기
    chunk_size : 한번에 buff에 저장하고, return할 chunk의 크기    
    """
    self.buff_size = buff_size
    self.chunk_size = chunk_size
    self.datasets_path = [root_path + dataset for dataset in dataset_name]
    self.dataloaders = [IterableDatasetCustom(path) for path in self.datasets_path]
    self.len_dataloaders = [len(loader) for loader in self.dataloaders]
    self.full_corpus_len = sum(self.len_dataloaders)
    self.iterator = [iter(dataloader) for dataloader in self.dataloaders]

    self.ratio_dataloader = self.makeMaskForDataLoader(self.len_dataloaders)

    self.buff = self.init_buff()
    print(f"sum of the length of corpus is {sum(len_ for len_ in self.len_dataloaders)}")

  def generate(self):
    rand_idxs = random.sample(range(len(self.buff)), self.chunk_size)
    dataloader_idx = self.selectDataloader(self.ratio_dataloader)
    iterator = self.iterator[dataloader_idx]

    self.buff, output = self.pop_multi(rand_idxs)

    chunk = self.make_chunk(iterator, dataloader_idx)
    if not chunk[0] : # iterator가 chunk_size만큼 데이터가 없을 경우
      len_False = len([corpus for corpus in chunk if not corpus]) # 부족한 코퍼스 수
      chunk = [corpus for corpus in chunk if corpus] # False를 비움
      
      dataloader_idx = self.selectDataloader(self.ratio_dataloader)
      iterator = self.iterator[dataloader_idx]
      additional_chunk = self.make_chunk(iterator, dataloader_idx, chunk_size = len_False)
      chunk.extend(additional_chunk)
      assert len(chunk) == self.chunk_size, f"chunk size : {len(chunk)} || origianl chunk size : {self.chunk_size}"

    self.buff.extend(chunk)
    return output 








  def makeMaskForDataLoader(self, len_list : list) -> list:
    result = []
    accum = 0
    sum_ = sum(len_list)
    for len_ in len_list:
        accum += len_
        result.append(accum)
    return [ratio/sum_ for ratio in result]

  def selectDataloader(self, ratio_list : list):
    seed = random.random()
    for idx, ratio in enumerate(ratio_list):
        if seed <= ratio:
            return idx

  def init_buff(self) :
    buff = []
    for num in range(self.buff_size//self.chunk_size):
      dataloader_idx = self.selectDataloader(self.ratio_dataloader)
      iterator = self.iterator[dataloader_idx]
      buff.extend(self.make_chunk(iterator))

    for num in range(self.buff_size%self.chunk_size):
      buff.append(next(iterator))
    
    assert len(buff) == self.buff_size, f"original buff size : {self.buff_size} || buff size : {len(buff)}"
    print("-"*30, "initializing the buff is over", "-"*30)
    return buff

  def pop_multi(self, idxs : list) -> tuple:
    '''
    input : 코퍼스 리스트, 뽑고 싶은 인덱스 리스트
    output : idxs가 지워진 리스트, idxs만 선택된 리스트
    '''
    assert max(idxs) <= len(self.buff) - 1,  "The maximum integer of index is larger than the length of the corpus"
    return [line for num, line in enumerate(self.buff) if num not in idxs], [line for num, line in enumerate(self.buff) if num in idxs]


  def make_chunk(self, dataloader, dataloader_idx = 1000, chunk_size = None):
    if chunk_size == None:
      chunk_size = self.chunk_size

    try :
      # chunk = [next(dataloader) for i in range(chunk_size)]
      chunk = []
      for i in range(chunk_size):
        chunk.append(next(dataloader))

    except StopIteration :
      print(f"Loading {self.dataloaders[dataloader_idx].dataset_name} is over")

      self.deleteDataloader(dataloader_idx)
      
      print("-"*30, "Dataloaders left are below", "-"*30)
      [print(loader.dataset_name) for loader in self.dataloaders]

      len_chunk_leftover = len(chunk)
      chunk = [False]*(chunk_size - len_chunk_leftover) + chunk

      for i in range(chunk_size - len_chunk_leftover):
        try:
          chunk.append(next(dataloader))
          chunk.pop(0)

        except StopIteration :
          return chunk

    return chunk

  def deleteDataloader(self, dataloader_idx):
    self.iterator.pop(dataloader_idx)
    self.dataloaders.pop(dataloader_idx)
    self.len_dataloaders.pop(dataloader_idx)
    self.ratio_dataloader = self.makeMaskForDataLoader(self.len_dataloaders)
    

  
  def __len__(self):
    return self.full_corpus_len











def main():
  len_leftover = False
  dataset_name = ["kowiki_shuffled.csv",  "petition.csv", "NIKL_NEWSPAPER_jaehee.csv", "namuwiki.csv",] #
  buff_size = 50000 
  chunk_size = 2000
  dataloader = chainMultiIterableDataset(root_path="G:/공유 드라이브/TobigsTextConf 141516/cleaned/", dataset_name = dataset_name, buff_size=buff_size, chunk_size=chunk_size)
  count = 0

  with tqdm(total = sum(dataloader.len_dataloaders)) as progress_bar:
    while True:
      if not len(dataloader.iterator) == 1:
        count+=1
        corpus = dataloader.generate()
        print(len(dataloader.iterator), end = "\r")
        progress_bar.update(chunk_size)
      elif len(dataloader.buff) != 0 :
        print("start extracting corpus from buff")
        corpus = dataloader.buff
        len_buff = len(corpus)
        dataloader.buff = []
        assert len(dataloader.buff) == 0, "the length of buff is not 0"
        print("complete extracting corpus from buff")
        progress_bar.update(chunk_size)
        
      else:
        iterator = dataloader.iterator[0] # 여기서 쭉 뽑도록 ㄱㄱ
        corpus = []
        while True:
          try :
            corpus.append(next(iterator))
          except StopIteration :
            len_leftover = len(corpus)
            break
        

        
      with open("C:/Users/호양이네/Desktop/ToBigBird/chained_nikl_namuwiki.csv", "a", encoding = "utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(corpus)

      if len_leftover:
        print(f"전체 크기 : {count*chunk_size + len_buff + len_leftover} | count*chunk_size : {count*chunk_size} | len_buff : {len_buff} | len_leftover : {len_leftover}")
        break


 



if __name__ == "__main__":
  main()