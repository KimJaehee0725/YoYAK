import csv
from tqdm import tqdm
import random
from krwordrank.sentence import summarize_with_sentences



class chainedLoader :
    def __init__(self, path = "C:/Users/호양이네/Desktop/ToBigBird/chained_nikl_namuwiki.csv") -> None:
        self.path = path
        self.len = 0
    
    def __iter__(self):
        with open(self.path, mode = "r", newline="", encoding = "utf-8") as f:
            reader = csv.reader(f)
            for num, line in enumerate(reader):
                self.len += 1
                yield list(line)

def loadStopwords(path = "G:/내 드라이브/04_프로젝트/ToBigBird/codes/data/preprocess/data_merge&GSG/stopwords.txt") :
    load = open(path, encoding = "utf-8")
    corpus = [line.replace("\n", "") for line in load.readlines()]
    return corpus

def selectBestSentences(line : list):
    stopwords = loadStopwords()
    penalty = lambda x:0 if (25 <= len(x) <= 80) else 1

    _, _, key_idxs = summarize_with_sentences(
        line,
        penalty=penalty,
        diversity=0.5,    # cosine 유사도 기준 핵심문장 간의 최소 거리
        num_keywords=100,
        num_keysents= round(len(line)*0.45), #45% 문장 갯수만 추출후 -> 반올림
        scaling = lambda x:1,
        verbose=False,
        stopwords=stopwords)

    masks = [random.random() for i in range(len(key_idxs))]
    result = [idx for idx, mask in zip(key_idxs, masks) if mask > 0.33]

    return result

def getIndxs(line : list):
    try: 
      result = selectBestSentences(line)
    except:
      masks = [random.random() for i in range(len(line))]
      result = [idx for idx, mask in enumerate(masks) if mask > 0.3]
    
    return result

def getCorpusIndxs(line : list):
  indxs = getIndxs(line)
  return (line, indxs) 
