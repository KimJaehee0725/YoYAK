import os
import re
import kss
from tqdm import tqdm
import pandas as pd


def load(file_path, file_name):
    with open(os.path.join(file_path, file_name), "r", encoding="utf-8") as f:
        return f.readlines()

def save(src, save_path, file_name):
    save_file_name = file_name + ".csv"
    src.to_csv(os.path.join(save_path, save_file_name), encoding="utf-8")


def normalize_text(text):
    """정규표현식"""
    text = re.sub(r"[^·가-힣\x00-\x7F]", " ", text)  # ·, 한글, ASCII코드 외 삭제
    text = re.sub(r"[\(\{\[][\S ]+[\)\}\]]", "", text)  # 괄호(){}[]와 내부 내용 삭제
    text = re.sub(r"[\(\{\[]\s*[\)\}\]]", "", text)  # 내용없는 괄호(){}[] 삭제
    text = re.sub(r"\s{2,}", " ", text)  # 공백 정규화
    return text.strip()

def sentence_split(line):
    tmp_result = []
    header = re.compile("<.+>")
    if re.search(header, line):
        return tmp_result
    
    try:
        sentences = kss.split_sentences(line)
    except IndexError:
        return tmp_result
    
    for sentence in sentences:
        sentence = normalize_text(sentence)
        if len(sentence) > 10:
            tmp_result.append(sentence)
    return tmp_result


def preprocessing(lines, file_name):
    result = pd.DataFrame([], columns=["length", "doc"])
    idx = 0
    temp = []
    flag = False
    for i, line in enumerate(tqdm(lines, desc="[Pre-processing {}]".format(file_name))):
        line = line.strip()

        if "</doc>" in line:
            flag = True
        if not line:
            continue
        if len(line) < 4:
            continue

        temp.extend(sentence_split(line))
        if flag:
            sentence_count = len(temp)
            if sentence_count >= 6:  # 믄장 개수 5개 이하 버리기
                result.loc[idx] = [sentence_count, temp]
                idx += 1
            flag = False
            temp = []
    
    return result


def main():
    file_path = "text/AI"
    save_path = "processed/AI"
    file_names = sorted(os.listdir(file_path))
    for i in range(len(file_names)):
        if i <= 52:
            file_names.pop(0)
    print(file_names)

    for i, file_name in enumerate(file_names):
        lines = load(file_path, file_name)
        lines_pp = preprocessing(lines, file_name)
        save(lines_pp, save_path, file_name)

if __name__ == '__main__':
    main()