from transformers import pipeline
import numpy as np

kobigbird_pipe = pipeline(
    "fill-mask",
    model='monologg/kobigbird-bert-base',
    tokenizer=('monologg/kobigbird-bert-base', {'use_fast':True}),
    framework = 'pt',
    top_k  = 1
    )



def mask_to_replace(masked_text):
  '''
  masked_text: [MASK]토큰 들어간 문장을 str type으로 입력받음
  ex) 안녕하세요 국민 여러분. 이번 대통령 선거 야당 후보는 문재인이고 여당 후보는 [MASK]입니다. 홍준표가 [MASK] 되면 좋겠습니다.

  return: [MASK]토큰을 다른 토큰으로 대체하여 str type으로 text 내보냄
  ex) 안녕하세요 국민 여러분. 이번 대통령 선거 야당 후보는 문재인이고 여당 후보는 홍준표입니다. 홍준표가 당선 되면 좋겠습니다.
  '''


  def one_mask(text):
    replaced = kobigbird_pipe(text)[0]['token_str']
    text = text.replace('[MASK]',replaced,1)
    return text


  text = masked_text
  mask = 0 

  if masked_text.count('MASK') == 1:
    return one_mask(text)

  elif masked_text.count('MASK') == 0:
    
    return text

  else:

    while True:

      pipe_result = kobigbird_pipe(text)

      if text.count('MASK') == 1:
        return one_mask(text)

      else:
        scoring = [pipe_result[i][0]['score'] for i in range(len(pipe_result))]

        highest_score_idx = np.argmax(scoring)
        text = pipe_result[highest_score_idx][0]['sequence']
        text = text.replace('[CLS]','').replace('[SEP]','').strip()

      if 'MASK' not in text:
        break

  return text



def load_kcbert():
  pass


#### 김재희가 짠 개똥같은 KcBert 코드
def mask_restore_infer_kcbert(masked_text) :
  from transformers import AutoTokenizer, AutoModelWithLMHead
  tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-large")
  model = AutoModelWithLMHead.from_pretrained("beomi/kcbert-large")

  # masked_text = clean(masked_text)
  splitted, last_tokens_start = split_corpus(tokenizer, masked_text)
  print(splitted)
  len_list = len(splitted)
  restored_list = []

  test = []
  for num, source in enumerate(splitted) :
    masked_inputs = tokenizer.encode(source)
    mask_idx = [num for num, idx in enumerate(masked_inputs) if idx == 4]
    restored_kcbert = fill_entire_mask(model, masked_inputs, mask_idx)
    test.append(restored_kcbert)
    if num == len_list - 1 :
      restored_kcbert = restored_kcbert[last_tokens_start:]
    restored_list.append(restored_kcbert)
  return tokenizer.decode(restored_list, skip_special_tokens = True)

def fill_entire_mask(model, input_ids, mask_idx, num = 1, is_gpu = False, device = None):
  
  assert count_mask(input_ids) > num, f"입력 문장의 마스크 토큰 수가 {num}보다 적습니다. {count_mask(input_ids)}"

  while count_mask(input_ids) != 0:
    if count_mask(input_ids) < num:
      
      num = count_mask(input_ids)
    logits = cal_logits(model, input_ids, is_gpu, device)
    top_k_idx, predicted_token_idx = find_topk_token(logits, mask_idx, num = num)
    mask_idx = delete_mask_idx(mask_idx, top_k_idx)
    input_ids = fill_mask_token(top_k_idx, predicted_token_idx, input_ids)

  return input_ids[0]

def cal_logits(model, input_ids, is_gpu = False, device = None):
  model = model.eval()
  if is_gpu :
    model = model.to(device) 
    input_ids = input_ids.to(device)
  logits = model(input_ids)
  return logits.logits

def find_topk_token(logits, mask_idx, num = 5):
  probs = torch.softmax(logits, dim = 1)
  mask_probs = torch.max(probs[mask_idx], axis = 1).values
  token_prob = dict(zip(mask_idx, mask_probs))
  sorted_mask_idx = sorted(token_prob.items(), key = lambda x : x[1], reverse = True)
  top_k_idx = [items[0] for number, items in enumerate(sorted_mask_idx) if number < num]
  predicted_token_idx = torch.argmax(probs[top_k_idx], dim = 1)

  return top_k_idx, predicted_token_idx

def delete_mask_idx(mask_idx, top_k_idx):
  save_idx_list = []
  for idx in mask_idx:
    if idx not in top_k_idx:
      save_idx_list.append(idx)
  return save_idx_list

def fill_mask_token(mask_token_idx, token_fill, input_ids):
  input_ids[0][mask_token_idx] = token_fill
  return input_ids

def count_mask(input_ids):
  return sum(input_ids == 4)

def clean(x):
  import re
  import emoji
  from soynlp.normalizer import repeat_normalize

  emojis = list({y for x in emoji.UNICODE_EMOJI.values() for y in x.keys()})
  emojis = ''.join(emojis)
  pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
  url_pattern = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

  x = pattern.sub(' ', x)
  x = url_pattern.sub('', x)
  x = x.strip()
  x = repeat_normalize(x, num_repeats=2)
  return x

def split_corpus(tokenizer, text, overlap = None, max_len = 298):
    tokens = tokenizer(text).input_ids
    tokens_len = len(tokens)
    split_list = []
    if overlap == None:
        list_len = tokens_len//max_len
        for i in range(list_len + 1) :
            if i == list_len:
                text_decode = tokenizer.decode(tokens[-max_len:], skip_special_tokens = False)
            else :
                text_decode = tokenizer.decode(tokens[i*max_len:(i+1)*max_len], skip_special_tokens = False)
            split_list.append(text_decode.replace("[SEP]", "")) #sep pad cls
    
    return split_list, max_len - (tokens_len%max_len)




