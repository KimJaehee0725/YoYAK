import json 
import regex as re

"""
Hiragana: Unicode: 3040-309F
Katakana: Unicode: 30A0–30FF
"""

def findJp(chr):
    hiragana_start = '\u3040'
    hiragana_end = '\u309F'
    katakana_start = '\u30A0'
    katakana_end = '\u30FF'
    others = '々〆〤'
    return  (hiragana_start < chr <  hiragana_end) or (katakana_start < chr < katakana_end) or (chr in others)

def replaceThings(line: str) :
    line = re.sub(r"[『』「」]|#[\S]+|", "", line) 
    line = re.sub(r"[^·가-힣\x00-\x7F]", " ", line)  # ·, 한글, ASCII코드 외 삭제
    line = re.sub(r"[\(\{\[][\S ]+[\)\}\]]", "", line)  # 괄호(){}[]와 내부 내용 삭제
    line = re.sub(r"\s'", " ' ", line)
    line = re.sub("\"", " \" ", line)
    line = re.sub(r"\s{2,}", " ", line)  # 공백 정규화
    return line

def calNotKoreanRatio(line : str) :
    regex = r"[^ㄱ-힣 ]+"
    full_len = len(line)
    not_kor_len = sum([len(word) for word in re.findall(regex, line)])

    return (not_kor_len/full_len) > 0.15 if full_len != 0 else True

def removeBracket(line : str) :
    regex = r".]"
    return re.sub(regex, ".", line)

def loadOneLine(line : str) :
    """
    input : json line
    output : False(if japanese is included in corpus) or
             list of paragraph(str) (ex. [paragraph1, paragraph2, ..., paragraphn])
    """
    line_load = json.loads(line)['text']

    for chr in line_load:
        is_Japanese = findJp(chr)
        if is_Japanese:
            return False, False

    line_return = [sentence for sentence in line_load.split('\n')]
    len_line = sum(len(sentence) for sentence in line_return)
    if not len_line == 0:
        return line_return, len_line
    else: 
        return False, False

def cleanLine(line : str) :
    line = replaceThings(line)
    line = removeBracket(line)
    if not calNotKoreanRatio(line) :
        return line

def cleanRawLines(
    source_str : list,
    min_line : int = 10, 
    max_total_len : int = 15000) :
    """
    input : list of paragraph
    output : None(if the length of every paragraph in corpus is shorter than min_line) or
             str as one corpus
    """

    line_cleaned = [sentence for sentence in source_str if len(sentence) > min_line]
    if len(line_cleaned) == 0 :
        return None, 0
    
    line_cleaned = [cleanLine(sentence) for sentence in line_cleaned]
    line_cleaned = [sentence for sentence in line_cleaned if sentence]
    line_cleaned = " ".join(line_cleaned)
    # line_cleaned = line_cleaned[:max_total_len]
    len_line = len(line_cleaned)
    return line_cleaned, len_line

def removeOver4096Sentence(corpus:list) ->list:
    def isOvertheLength(line:str):
        if len(line)>3000: # 3000자가 4096 토큰을 의미하지 않지만 이 정도면 충분히 비정상적인 양
            return False
        else:
            return True
    result = [sentence for sentence in corpus if isOvertheLength(sentence)]
    return result


# def makeSentenceLengthUnder4096 (tokenizer, corpus:list) -> list :
#     """
#     input : list of sentences of tokens
#     output : list of sentences that is converted to string and has length under 4096
#     """
#     result = []
#     total_len = 0

#     for sentence in corpus:
#         len_sentence = len(sentence)
#         if len_sentence > 10 :

#             if total_len + len_sentence < 4096:
#                 result.append(sentence)
#                 total_len += len_sentence
        
#             else: 
#                 break
            
#     if not len(sentence) == 0: # 코퍼스 내 문장길이가 모두 10보다 짧으면 문장이 다 지워지는 경우가 발생
#         return [tokenizer.convert_tokens_to_string(sentence) for sentence in result], total_len
#     else :
#         return False, False


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

    result = [(corpus, len_) for corpus, len_ in zip(result_corpus, result_len)]
    return result

