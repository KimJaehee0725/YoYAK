import re
import csv
import pandas as pd
import random
from kss import split_sentences
from konlpy.tag import Okt
from transformers import AutoTokenizer
from p_tqdm import p_umap
from tqdm import tqdm
from multiprocessing import freeze_support
import warnings
warnings.filterwarnings("ignore")



tokenizer = Okt()
def cleaning(text):
    text = re.sub(r"(ftp:\/\/|www\.|https?:\/\/){1}[a-zA-Z0-9u00a1-\uffff0-]{2,}\.[a-zA-Z0-9u00a1-\uffff0-]{2,}(\S*)", "", text) # url 제거
    text = re.sub(r"[^·가-힣\x00-\x7F]", " ", text)  # ·, 한글, ASCII코드 외 삭제
    text = re.sub(r"[\(\{\[][^)]*[\)\}\]]", "", text)  # 괄호(){}[]와 내부 내용 삭제
    text = re.sub(r"\s{2,}", " ", text)  # 공백 정규화
    text = re.sub(r"\\n", "", text) 
    text = re.sub(r"[=-]", "", text)
    text = re.sub(';', '',text)
    text = re.sub(r'!+', '!',text) # 다중 기호 제거
    text = re.sub(r'\?+', '?',text) # 다중 기호 제거
    text = re.sub(r',+', ',',text) # 다중 기호 제거
    text = re.sub(r'\.+', '.',text) # 다중 기호 제거
    text = re.sub("\\\"", "", text)
    text = re.sub("\\'", "'", text)
    return text

def makeSentenceLengthUnder4096 (tokenizer, corpus_tokenized:list, corpus_splitted:list) -> list :
    """
    input : list of sentences of tokens
    output : list of sentences that is converted to string and has length under 4096
    """
    result_corpus = []
    result_len = []

    total_len = 0
    corpus_under_4096 = []

    reserve = ["x", "x"]
    reserve_len = [0, 0]


    for tokenized, splitted in zip(corpus_tokenized, corpus_splitted):
        len_sentence = len(tokenized)
        if len_sentence > 10 :

            if total_len + len_sentence < 4096:
                reserve.append(splitted)
                reserve.pop(0)
                reserve_len.append(len_sentence)
                reserve_len.pop(0)
                corpus_under_4096.append(splitted)
                total_len += len_sentence
        
            else: 
                result_corpus.append(corpus_under_4096)
                result_len.append(total_len)
                corpus_under_4096 = []
                
                corpus_under_4096 = reserve.copy()
                corpus_under_4096.append(splitted)
                reserve.append(splitted)
                reserve.pop(0)
                reserve_len.append(len_sentence)
                total_len = sum(reserve_len)
                reserve_len.pop(0)
                
    if total_len < 4096:
        result_corpus.append(corpus_under_4096)
        result_len.append(total_len)

    # result = [(corpus, len_) for corpus, len_ in zip(result_corpus, result_len)]
    return result_corpus, result_len

def calNotKoreanRatio(line : str) :
    regex = r"[^ㄱ-힣 ]+"
    full_len = len(line)
    not_kor_len = sum([len(word) for word in re.findall(regex, line)])

    return (not_kor_len/full_len) > 0.15 if full_len != 0 else True

def preprocess_petition(content : str, tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")) :
    content_cleaned = cleaning(content)
    if calNotKoreanRatio(content_cleaned):
        return [False], [False]
    if len(content_cleaned) < 100 : # 한 코퍼스 당 최소 100글자는 되어야 함. 
        return [False], [False]
    content_splitted = split_sentences(content_cleaned)
    if len(content_splitted) < 12: # 한 코퍼스 당 최소 12 문장은 있어야 함. 
        return [False], [False]
    content_tokenized = [tokenizer.tokenize(sentence) for sentence in content_splitted]
    content_under_4096, content_len = makeSentenceLengthUnder4096(tokenizer, content_tokenized, content_splitted)

    return content_under_4096, content_len
    
all_df = pd.read_csv("G:/공유 드라이브/TobigsTextConf 141516/raw/petition.csv")

tokenizer = Okt()

all_df["corpus"] = all_df["title"] + all_df["content"]
corpus = list(all_df["corpus"])
corpus = [content for content in corpus if type(content) == str] # nan이 하나 존재

def main():
    print("preprocessing start")
    preprocess_result = p_umap(preprocess_petition, corpus)
    print("extracting corpus start")
    preprocess_corpus = [sentence for line, _ in tqdm(preprocess_result) for sentence in line if sentence]
    preprocess_corpus = [sentence for sentence in preprocess_corpus if len(sentence) > 11]
    print("extracting len start")
    preprocess_len = [str(len_) for _, len_list in tqdm(preprocess_result) for len_ in len_list if len_]

    random.shuffle(preprocess_corpus)
    with open("G:/공유 드라이브/TobigsTextConf 141516/cleaned/petition.csv", mode = "w", encoding = "utf-8", newline = "") as f:
        writer = csv.writer(f)
        writer.writerows(preprocess_corpus)
    with open("G:/공유 드라이브/TobigsTextConf 141516/cleaned/petition_log.csv", mode = "w", encoding = "utf-8", newline = "") as f: 
        writer = csv.writer(f)
        writer.writerows(preprocess_len)


if __name__ == "__main__":

    main()