import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from make_longformer import *
from transformers import PreTrainedTokenizerFast

# Dataset이 넘겨줘야할 것들
# encoder_inputs : token_ids, attention_mask
# decoder_inputs : token_ids, attention_mask
# label : token_ids 

# To Do List
# 1. 재희형이 만든 데이터로더랑 모델 연결
# 2. validation step과 test step에 rouge score 계산하는 부분 추가
# 3. Optimizer select learning rate scheduler 추가 > Done!!
# - layer normalization parameter도 학습시키는게 맞을까? > No! 우선은 layer normalization parameter도 학습하자
# - 이미 kobart 자체가 요약 데이터셋으로 조금만 fine-tuning해도 성능이 잘 나와서 1 epoch만 돌려도 충분할 거 같음)
# - learning_rate 값, warming_up step을 언제까지로 할지
# 4. gradient accumulating, clipping, amp backend, 같은 technical한 부분 추가할지? >> 어느정도 배치 올라가면 accumulating은 하지 말자..
# 5. Checkpoint 불러올 때 lightening binary > pytorch binary 변환 작업해주는 script file 추가 > Done!
# 6. inferene pipeline 만들기 > 작업중  .. gpu에 올려서 inference하도록 코드 변경하면 될듯??

class LongformerSummaryModule(pl.LightningDataModule):
    def __init__(self, train_file:str, valid_file:str, test_file:str, tokenizer_path:str, batch_size: int=8, num_workers: int=5):
        super().__init__()
        self.batch_size = batch_size
        self.train_file_path = train_file
        self.valid_file_path = valid_file
        self.test_file_path = test_file
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.num_workers = num_workers

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = iterableDataset(self.train_file_path)
        self.valid = iterableDataset(self.valid_file_path)
        self.test = iterableDataset(self.test_file_path)

    def train_dataloader(self):
        train = DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.test, batch_size=self.batch_size,  num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return test


class LongformerKobart(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = LongformerBartForConditionalGeneration.from_pretrained(hparams.model_path)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.hparams.tokenizer_path)
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token_id = 3
        self.ignore_token_id= -100

    def forward(self, inputs):
        encoder_inputs, decoder_inputs, labels = inputs

        return self.model(input_ids=encoder_inputs['token_ids'],
                          attention_mask=encoder_inputs['attention_mask'],
                          decoder_input_ids=decoder_inputs['token_ids'],
                          decoder_attention_mask=decoder_inputs['attention_mask'],
                          labels=labels['token_ids'], return_dict=True)
    
    # Use this method to caculate rouge score and inference
    def generate(self,inputs):
        encoder_inputs, decoder_inputs, labels = inputs
        return self.model.generate(input_ids=encoder_inputs['input_ids'], 
                                    attention_mask=encoder_inputs['attention_mask'],
                                    max_length=self.args.max_output_len,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    num_beams=1), labels['token_ids']


    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        return (loss)

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)
    
    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    # 수렴 이상하면 learning rate scheduler 추가하기
    def configure_optimizers(self):
        # Prepare optimizer
        '''

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        '''
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr, correct_bias=True)
        num_steps = self.hparams.dataset_size * self.hparams.max_epochs / self.hparams.gpus / self.hparams.batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup, num_training_steps=num_steps)
        lr_scheduler = {'scheduler': scheduler,  'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optimizer], [lr_scheduler] 

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--train_file', type=str, default='/ /train.csv', help='train file')
        parser.add_argument('--valid_file', type=str, default='/ /valid.csv', help='valid file')
        parser.add_argument('--test_file', type=str, default='/ /test.csv', help='test file')
        
        parser.add_argument("--model_path", type=str, default='longformer_kobart_initial_ckpt', help="Path to the checkpoint directory or model name")
        parser.add_argument("--tokenizer_path", type=str, default='longformer_kobart_initial_ckpt')
        
        parser.add_argument("--gpus", type=int, default=1, help="Number of gpus. 0 for CPU")
        parser.add_argument("--num_workers", type=int, default=5, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument("--max_epochs", type=int, default=2, help="Number of epochs")
        
        parser.add_argument('--lr',type=float,default=1e-7,help='The initial learning rate')
        parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        
        parser.add_argument("--max_output_len", type=int, default=1024, help="maximum num of output length. Used for training and testing")
        parser.add_argument("--max_input_len", type=int, default=4096, help="maximum num of input length. Used for training and testing")
        
        parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Number of training steps between checkpoints")

        parser.add_argument("--test", action='store_true', help="Test only, no training")
        return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dm = LongformerSummaryModule(args.train_file, args.valid_file, args.test_file, args.tokenizer_path,
                        batch_size=args.batch_size,
                        max_input_len=args.max_input_len,
                        max_output_len=args.max_output_len,
                        num_workers=args.num_workers)

    model = LongformerKobart(args)

    # args.default_root_dir : logs
    wandb_logger = WandbLogger(project='longformer_kobart',name=f'Longformer_testing')
    tb_logger = TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))

    checkpoint_callback = ModelCheckpoint(dirpath=args.default_root_dir, filename='model_chp/{epoch:02d}-{val_loss:.3f}', every_n_train_steps=args.checkpoint_interval, save_top_k=-1, verbose=True, monitor='val_loss', mode='min', save_last=True)
    args.dataset_size = 203037  # hardcode dataset size. Needed to compute number of steps for the lr scheduler
    print(args)

    trainer = pl.Trainer(gpus=args.gpus, 
                        distributed_backend='ddp' if torch.cuda.is_available() else None,
                        accumulate_grad_batches = 1 # if 4, 4 batch > 16 batch, 이거 숫자 늘리면 learning_rate scheduler parameter 조정 필요
                        max_epochs=args.max_epochs,
                        val_check_interval= 0.2, # check validation set 4 times during a training epoch
                        check_val_every_n_epoch=1, 
                        logger=[wandb_logger, tb_logger],
                        callbacks=checkpoint_callback,
                        logger=tb_logger,
                        gradient_clip_val=0.0 # No gradient clipping,
                        log_every_n_steps=50 # logging frequency in training step,
                        max_epochs = args.max_epochs)
    if not args.test:
        trainer.fit(model,dm)
    trainer.test(model)

if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="longformer_kobart")
    parser = LongformerKobart.add_model_specific_args(main_arg_parser)
    args = parser.parse_args()
    main(args)