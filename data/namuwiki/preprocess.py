
from util_clean import loadOneLine, cleanRawLines, makeSentenceLengthUnder4096, removeOver4096Sentence
import csv
import time
from tqdm import tqdm
from multiprocessing import Pool, freeze_support
from transformers import AutoTokenizer
from kss import split_sentences

def split_sentences_safe(corpus):
    return split_sentences(corpus, safe =  True)

def tokenizeByLine(
    p = None,
    root_dir = "C:/Users/호양이네/Desktop/ToBigBird/new",
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base"),
    min_corpus = 100,
    min_line = 100,
    min_sentence = 12,
    chunk = 50000,
    pass_line = -1) :
    """
    min_corpus : 한 문서 최소 문자 수
    min_line : 한 문단 최소 문자 수
    min_sentence : 한 문서내 최소 문장 수
    chunk : 한번에 불러올 코퍼스 수
    pass_line : 중간부터 하고 싶을경우 뛰어넘을 코퍼스 수
    """

    parsed_path = root_dir + "/namuwiki_20210301_parsed.json"
    print(parsed_path)
    save_path = root_dir + "/namuwiki_cleaned.csv"
    len_path = root_dir + "/len_log.txt"
    log_path = root_dir + "/file_log.txt"
    
    with open(parsed_path)  as f_parsed:
        total_len_loaded = 0
        total_len_cleaned = 0
        total_len_tokenized = 0
        
        total_time = 0

        count_cleaned = 0
        count_tokenized = 0
        

        chunk_parsed = []

        with tqdm(total = 100) as pbar:
            for num, line in enumerate(f_parsed) :
                if num < pass_line:
                    if num%5000 == 0:
                        print(f"{num} passed")
                    pass

                if num%chunk == 0:
                    time_start = time.time()

                if not (num + 1)%chunk == 0:
                    chunk_parsed.append(line)
                    
                else:
                    print("\n")   
                    chunk_loaded = [loadOneLine(corpus) for corpus in chunk_parsed if chunk_parsed]
                    time_load = time.time()
                    len_loaded = [len_ for _, len_ in chunk_loaded if len_]


                    chunk_len_cleaned = [cleanRawLines(corpus, min_line = min_line) for corpus, len_loaded in chunk_loaded if len_loaded > min_corpus]
                    
                    len_cleaned = [len_ for _, len_, in chunk_len_cleaned]
                    chunk_cleaned = [corpus for corpus, len_ in chunk_len_cleaned if not len_ == 0]
                    
                    chunk_splitted = [*p.map(split_sentences_safe, chunk_cleaned)]

                    chunk_splitted = [removeOver4096Sentence(corpus) for corpus in chunk_splitted]
                    chunk_splitted = [corpus for corpus in chunk_splitted if len(corpus) >= min_sentence]
                    
                    
                    time_clean_split = time.time()

                    chunk_tokenized = [[tokenizer.tokenize(sentence) for sentence in corpus] for corpus in chunk_splitted]
                    time_tokenize = time.time()

                    chunk_len_corpus = [makeSentenceLengthUnder4096(corpus_tokenized = tokenized, corpus_splitted = splitted, tokenizer = tokenizer) for tokenized, splitted in zip(chunk_tokenized, chunk_splitted)]
                    
                    chunk_len_corpus = [corpus for chunk in chunk_len_corpus for corpus in chunk]

                    len_tokenized = [len_ for _, len_ in chunk_len_corpus if len_] # 코퍼스 내 문장이 없는 경우 있음
                    chunk_corpus = [corpus for corpus, _ in chunk_len_corpus if corpus]
                    time_afterprocess = time.time()
                    
                    total_len_loaded += sum(len_loaded)
                    total_len_cleaned += sum(len_cleaned)
                    total_len_tokenized += sum(len_tokenized)

                    count_cleaned += len(len_cleaned)
                    count_tokenized += len(len_tokenized)

                    with open(save_path, mode = "a", newline = "", encoding = "utf-8") as f_tokenized:
                        writer = csv.writer(f_tokenized)
                        writer.writerows(chunk_corpus)

                    with open(len_path, 'a', encoding = "utf-8") as f_len:
                        [f_len.write(str(len_) + '\n') for len_ in len_tokenized]

                    chunk_parsed = []
                    time_end = time.time()
                    pbar.update(chunk/8650)

                    total_time = time_end - time_start

                    log = f"""전체 {num+1}개 문서 중, | 정제된 문서 수 : {count_cleaned} | 최종 문서 수 : {count_tokenized}
{'문서 평균 길이 |':>15} 파싱 : {round(total_len_loaded/num,1):>11} | 정제 : {round(total_len_cleaned/count_cleaned,1):>11} | 토크나이징 : {round(total_len_tokenized/count_tokenized,1):>11}
{'작업 소요 시간 |':>15} 전체 : {round(total_time, 3):>11} | 로드 : {round(time_load - time_start, 3):>11} | 전처리 : {round(time_clean_split - time_load, 3):>11} | 토크나이징 : {round(time_tokenize - time_clean_split, 3):>11} | 후처리 : {round(time_afterprocess - time_tokenize, 3):>11}
------------------------------------------------------------------------------------------------------------------------------------------"""
                    print(log)
                    with open(log_path, "a", encoding="utf-8") as f_log:
                        f_log.write(log + "\n")



def main():
    p = Pool(7)
    tokenizeByLine(p = p)

if __name__ == "__main__":
    freeze_support()
    main()










