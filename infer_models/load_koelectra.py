from transformers import ElectraTokenizer, ElectraForPreTraining
import numpy as np
import regex as re
import torch

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
discriminator = ElectraForPreTraining.from_pretrained("monologg/koelectra-base-v3-discriminator")

def discriminator_replaced_token(fake_sentence, threshold):
  # 문장과 임계치가 주어졌을 때, replaced token을 나타낸 tf_index와 index 반환
    if type(fake_sentence) == str:
        fake_tokens = tokenizer.tokenize(fake_sentence)
    else:
        fake_tokens = fake_sentence
    length = len(fake_tokens)
    fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
    discriminator_outputs = discriminator(fake_inputs)
    predictions = torch.sigmoid(discriminator_outputs[0])
    # true, false list 생성
    tf_index = list(map(lambda x: True if x > threshold else False, predictions[0][1:-1]))
    # 해당 토큰의 인덱스 추출
    index = list(filter(lambda x: predictions[0][1:-1][x] > threshold, list(range(length))))
    np_fake_tokens = np.array(fake_tokens)
    return tf_index, index

def token_to_sentence(masked_token):
    #토큰을 다시 문장을 변환
    masked_token = ' '.join(masked_token).replace(" ##","")
    masked_token =  re.sub(r'(\s)([-=+,#/\?:^.@*\"※~ㆍ!%‘|\(\)\[\]`\'…》\”\“\’·])(\s)', '\\2', masked_token)
    masked_token = re.sub(r'(\.)([가-힣{}])','\\1 \\2', masked_token)
    masked_token = re.sub(r'([가-힣]) (\.)','\\1\\2', masked_token)
    masked_token = re.sub("{}", "[MASK]", masked_token)
    return masked_token

def discriminator_replaced_token_prediction(fake_sentence):
    # 문장과 임계치가 주어졌을 때, prediction값 반환
    if type(fake_sentence) == str:
        fake_tokens = tokenizer.tokenize(fake_sentence)
    else:
        fake_tokens = fake_sentence
    length = len(fake_tokens)
    fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
    discriminator_outputs = discriminator(fake_inputs)
    predictions = torch.sigmoid(discriminator_outputs[0])
    return predictions[0][1:-1]

def mask_per_510(fake_sentence, threshold):
    # 문장을 510토큰을 기준으로 잘라서 koelectra에 넣음. replaced token을 출력하고, 이를 {}로 표기하여 문장 반환
    total_fake_tokens = tokenizer.tokenize(fake_sentence)
    masked_token_index = []
    for i in range((len(total_fake_tokens)//510) +1):
        tf_index, index = discriminator_replaced_token(total_fake_tokens[i*510:(i+1)*510], threshold)
        masked_token_index.extend(tf_index)
    np_fake_tokens = np.array(total_fake_tokens)
    print(np_fake_tokens[masked_token_index])
    np_fake_tokens[masked_token_index] = '{}'
    #return tokenizer.decode(np_fake_tokens)
    return token_to_sentence(np_fake_tokens)

def mask_overlap_concat(fake_sentence, threshold):
    # 510토큰을 기준으로 앞 뒤로 겹치게 잘라서 koelectra에 넣음.
    # 앞은 그대로, 뒤는 겹치지 않은 부분만 concat.
    # replaced token을 출력하고, 이를 {}로 표기하여 문장 반환
    total_fake_tokens = tokenizer.tokenize(fake_sentence)
    length_token = len(total_fake_tokens)
    masked_token_tf_index = []
    replaced_token = []
    np_fake_tokens = np.array(total_fake_tokens)
    n_part = (length_token//510)+1
    if n_part == 1:
        overlap_length = 0
    else:
        overlap_length = (n_part*510-length_token)//(n_part-1)
        overlap_length_f = overlap_length + (n_part*510-length_token)%(n_part-1)
    for i in range(n_part):
        start_point = 0
        if i == 0:
            if n_part == 1:
                end_point = length_token
                l_point = length_token
            else:
                l_point = 510
                end_point = 510
        elif i == n_part-1:
            l_point = length_token-end_point
            start_point = i*510 - (i-1)*overlap_length - overlap_length_f
            end_point = length_token
        else:
            l_point = (510-overlap_length)
            start_point = i*510-i*overlap_length
            end_point = start_point+510
        tf_index, index = discriminator_replaced_token(total_fake_tokens[start_point:end_point], threshold)  
        masked_token_tf_index.extend(tf_index[-l_point:])

    print(np_fake_tokens[masked_token_tf_index])
    np_fake_tokens[masked_token_tf_index] = '{}'

    return token_to_sentence(np_fake_tokens)

def mask_overlap_average(fake_sentence, threshold):
    # 510토큰을 기준으로 앞 뒤로 겹치게 잘라서 koelectra에 넣음.
    # 중간의 logit은 앞 뒤의 logit을 평균내서 사용
    # replaced token을 출력하고, 이를 {}로 표기하여 문장 반환
    total_fake_tokens = tokenizer.tokenize(fake_sentence)
    length_token = len(total_fake_tokens)
    n_part = (length_token//510)+1
    if n_part == 1:
        overlap_length = 0
    else:
        overlap_length = (n_part*510-length_token)//(n_part-1)
        overlap_length_f = overlap_length + (n_part*510-length_token)%(n_part-1)

    masked_token = []
    np_fake_tokens = np.array(total_fake_tokens)

    for i in range(n_part):
        if i == 0:
            if n_part == 1:
                predictions = discriminator_replaced_token_prediction(total_fake_tokens[:length_token])
            else:
                pred = discriminator_replaced_token_prediction(total_fake_tokens[:510])
                pred_ = pred[:510-overlap_length]
                pred_0 = pred[-overlap_length:]
                predictions = pred_

        elif i == n_part-1:
            pred = discriminator_replaced_token_prediction(total_fake_tokens[i*510 - (i-1)*overlap_length - overlap_length_f:])
            pred_ = pred[overlap_length_f:]
            pred_0 = (pred[:overlap_length_f] + pred_0)/2  
            predictions = torch.cat((predictions,pred_0, pred_),0)
                          
        else:
            pred = discriminator_replaced_token_prediction(total_fake_tokens[i*510-i*overlap_length:i*510-i*overlap_length+510])
            if i == n_part-2:
                pred_0 = (pred[:overlap_length] + pred_0)/2
                pred_ = pred[overlap_length:510-overlap_length_f]
                predictions = torch.cat((predictions,pred_0,pred_),0)
                pred_0 = pred[-overlap_length_f:]
            else:
                pred_0 = (pred[:overlap_length] + pred_0)/2
                pred_ = pred[overlap_length:510-overlap_length]
                predictions = torch.cat((predictions,pred_0,pred_),0)
                pred_0 = pred[-overlap_length:]
 
    tf_index = list(map(lambda x: True if x > threshold else False, predictions))
    print(np_fake_tokens[tf_index])
    np_fake_tokens[tf_index] = '[MASK]'

    return token_to_sentence(np_fake_tokens)