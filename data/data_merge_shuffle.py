from torch.utils.data import IterableDataset
import csv
import random

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

class IterableDatasetCustom(IterableDataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = data_path.split("/")[-1]
        self.len = 0

        # 데이터셋 전체 갯수 세기 위한 코드... 필요없을 것 같긴 한데 일단 ㄱ 
        print(f"Started Counting {self.dataset_name}")

        with open(data_path, mode = "r", newline="", encoding = "utf-8-sig") as dataset:
            reader = csv.reader(dataset)
            for num, line in enumerate(reader):
                # corpus = list(line)
                # corpus_len = sum([len(tokenizer(sentence)) for sentence in corpus])
                self.len += 1

                if num%1000 == 0:
                    print(f"{num} Corpus was read", end = "\r")

        print(f"Finished Counting {self.dataset_name}")
        print(f"The Length of {self.dataset_name} dataset is {self.len}")
        print("-"*50)

    
    def __iter__(self):
        with open(self.data_path, mode = "r", newline="", encoding = "utf-8-sig") as dataset:
            reader = csv.reader(dataset)
            for line in reader:
                yield list(line)
    
    def __len__(self):
        return self.len

class chainMultiIterableDataset(IterableDataset):
  def __init__(self, root_path, dataset_name, buff_size = 100):
    super().__init__()
    self.buff_size = buff_size
    self.datasets_path = [root_path + dataset for dataset in dataset_name]
    self.dataloaders = [IterableDatasetCustom(path) for path in self.datasets_path]
    self.len_dataloaders = [len(loader) for loader in self.dataloaders]
    self.iterator = [iter(dataloader) for dataloader in self.dataloaders]

    self.ratio_dataloader = self.makeMaskForDataLoader(self.len_dataloaders)

    self.buff = self.init_buff()

  def generate(self):
    rand_idx = random.randint(0, self.buff_size)
    dataloader_idx = self.selectDataloader(self.ratio_dataloader)
    iterator = self.iterator[dataloader_idx]
    
    try : 
      self.buff.append(next(iterator))

    except StopIteration:
      print(f"Loading {self.dataloaders[dataloader_idx].dataset_name} is over")
      self.iterator.pop(dataloader_idx)
      self.len_dataloaders.pop(dataloader_idx)
      self.ratio_dataloader = self.makeMaskForDataLoader(self.len_dataloaders)

      dataloader_idx = self.selectDataloader(self.ratio_dataloader)
      iterator = self.iterator[dataloader_idx]
      self.buff.append(next(iterator))
    
    rand_output=  self.buff.pop(rand_idx)

    return rand_output 

  def makeMaskForDataLoader(self, len_list : list) -> list:
    result = []
    accum = 0
    for len_ in len_list:
        accum += len_
        result.append(accum)
    return result

  def selectDataloader(self, ratio_list : list):
    seed = random.random()
    for idx, ratio in enumerate(ratio_list):
        if seed <= ratio:
            return idx

  def init_buff(self):
    buff = []
    for num in range(self.buff_size-1):
      dataloader_idx = self.selectDataloader(self.ratio_dataloader)
      iterator = self.iterator[dataloader_idx]
      buff.append(next(iterator))
    return buff

test_dataloader = chainMultiIterableDataset(root_path="G:/공유 드라이브/TobigsTextConf 141516/cleaned/", dataset_name = ["NIKL_NEWSPAPER.csv", "namuwiki_cleaned.csv"])
# test_iter = iter(test_dataloader)
for i in range(5):
  print("-"*50)
  print(test_dataloader.generate())