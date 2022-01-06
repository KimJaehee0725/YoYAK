import transformers
transformers.logging.set_verbosity_error()

from transformers import pipeline

kobigbird_pipe = pipeline(
    "fill-mask",
    model='monologg/kobigbird-bert-base',
    tokenizer=('monologg/kobigbird-bert-base', {'use_fast':True}),
    framework = 'pt',
    top_k  = 1
    )



def mask_restore_infer(masked_text):
  '''
  masked_text: [MASK]토큰 들어간 문장을 str type으로 입력받음
  ex) 안녕하세요 국민 여러분. 이번 대통령 선거 야당 후보는 문재인이고 여당 후보는 [MASK]입니다. 홍준표가 [MASK] 되면 좋겠습니다.

  return: [MASK]토큰을 다른 토큰으로 대체하여 str type으로 text 내보냄
  ex) 안녕하세요 국민 여러분. 이번 대통령 선거 야당 후보는 문재인이고 여당 후보는 홍준표입니다. 홍준표가 당선 되면 좋겠습니다.
  '''

  text = masked_text
  mask = 0 

  if masked_text.count('MASK') == 1:
    replaced = kobigbird_pipe(masked_text)[0]['token_str']
    text = text.replace('[MASK]',replaced,1)

    return text

  else:
    while True:
      replaced = kobigbird_pipe(masked_text)[mask][0]['token_str']
      text = text.replace('[MASK]',replaced,1)

      mask += 1

      if 'MASK' not in text:
        break

  return text