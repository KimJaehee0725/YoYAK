'''
pytorch-ligtening binary에서 pytorch binary file로 변환하는 코드
해당 script 파일 실행 후, longformer_kobart_trained_ckpt에 훈련된 모델 chekpoint가 저장됨 (inference 시에 해당 ckpt 사용)
'''

import argparse
from train import LongformerKobart
from make_longformer import *

import yaml
parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None, type=str) # logs/tb_logs/default/version_0/hparams.yaml
parser.add_argument("--model_binary", default=None, type=str) # logs/model_chp/*.ckpt
parser.add_argument("--output_dir", default='longformer_kobart_trained_ckpt', type=str) # pytorch binary가 저장되는 경로
args = parser.parse_args()

with open(args.hparams) as f:
    hparams = yaml.load(f)
    
inf = LongformerKobart.load_from_checkpoint(args.model_binary, hparams=hparams)

inf.model.save_pretrained(args.output_dir)