import os
import argparse
import random
from re import L
import numpy as np
from pytorch_lightning.accelerators import accelerator

import torch
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from model.make_longformer import *
from loader_map_style import *
from transformers import PreTrainedTokenizerFast

import logging
logging.getLogger("lightning").setLevel(logging.CRITICAL)

# Dataset이 넘겨줘야할 것들
# encoder_inputs : token_ids, attention_mask
# decoder_inputs : token_ids, attention_mask
# label : token_ids 

# To Do List
# 1. 재희형이 만든 데이터로더랑 모델 연결 > Done!
# 2. validation step과 test step에 rouge score 계산하는 부분 추가
# 3. Optimizer select learning rate scheduler 추가 > Done!
# - layer normalization parameter도 학습시키는게 맞을까? > Yes! 우선은 layer normalization parameter도 학습하자
# - 이미 kobart 자체가 요약 데이터셋으로 조금만 fine-tuning해도 성능이 잘 나와서 1 epoch만 돌려도 충분할 거 같음)
# - learning_rate 값, warming_up step을 언제까지로 할지
# 4. gradient accumulating, clipping, amp backend, 같은 technical한 부분 추가할지? >> 어느정도 배치 올라가면 accumulating은 하지 말자.. > Done!
# 5. Checkpoint 불러올 때 lightening binary > pytorch binary 변환 작업해주는 script file 추가 > Done!
# 6. inferene pipeline 만들기 > 작업중  .. gpu에 올려서 inference하도록 코드 변경하면 될듯??
# 7. model.generate hyperparameter 조정 : num_beams, n_gram 중복, 등등.. 

class LongformerSummaryModule(pl.LightningDataModule):
    def __init__(self, args:dict):
        super().__init__()
        self.batch_size = args.batch_size
        self.train_file_path = args.train_file
        self.valid_file_path = args.valid_file
        self.test_file_path = args.test_file
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
        self.num_workers = args.num_workers

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = ToBigsDataset(self.train_file_path)
        self.valid = ToBigsDataset(self.valid_file_path)
        self.test = ToBigsDataset(self.test_file_path)

    def train_dataloader(self):
        train = DataLoader(self.train, batch_size = self.batch_size, collate_fn = collat_batch, drop_last = True)

    def val_dataloader(self):
        valid = DataLoader(self.valid, batch_size = self.batch_size, collate_fn = collat_batch, drop_last = True)
        return valid

    def test_dataloader(self):
        test = DataLoader(self.test, batch_size = self.batch_size,collate_fn = collat_batch, drop_last = True)
        return test


class LongformerKobart(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = LongformerBartForConditionalGeneration.from_pretrained(self.hparams.model_path)
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
                          labels=labels['token_ids'])


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
        outs = self(batch)
        loss = outs['loss']
        return (loss)

    def test_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('test_loss', torch.stack(losses).mean(), prog_bar=True)

    def configure_optimizers(self):
        # Prepare optimizer
        '''
        # code for not update layer normalization parameters
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
        num_steps = int(self.hparams.dataset_size * self.hparams.max_epochs / self.hparams.gpus / self.hparams.batch_size)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup, num_training_steps=num_steps)
        lr_scheduler = {'scheduler': scheduler,  'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optimizer], [lr_scheduler] 

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--train_file', type=str, default='./data/toy_data/toy_train.csv', help='train file path') # toy > final 경로변경
        parser.add_argument('--valid_file', type=str, default='./data/toy_data/toy_valid.csv', help='valid file path') # toy > final 경로변경
        parser.add_argument('--test_file', type=str, default='./data/toy_data/toy_test.csv', help='test file path') # toy > final 경로변경
        
        parser.add_argument("--model_path", type=str, default='./model/longformer_kobart_initial_ckpt', help="Path to the checkpoint directory or model name")
        parser.add_argument("--tokenizer_path", type=str, default='./model/longformer_kobart_initial_ckpt')
        parser.add_argument("--default_root_dir", type=str, default='logs', help="parent directory of log files")
        
        parser.add_argument("--gpus", type=int, default=4, help="Number of gpus. 0 for CPU")
        parser.add_argument("--num_workers", type=int, default=5, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size") 
        parser.add_argument("--max_epochs", type=int, default=5, help="Number of epochs") # epoch 지정 해주기!
        
        parser.add_argument('--lr',type=float,default=0.00003,help='The initial learning rate')
        parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps") #epoch batch_size, gpu 개수에 따라 조정해줘야함!
        
        parser.add_argument("--max_output_len", type=int, default=1024, help="maximum num of output length. Used for training and testing")
        parser.add_argument("--max_input_len", type=int, default=4096, help="maximum num of input length. Used for training and testing")
        
        parser.add_argument("--checkpoint_interval", type=int, default=300000, help="Number of training steps between checkpoints") # 우선읜 데이터셋의 1/10으로 해놓음

        parser.add_argument("--test", action='store_true', help="Test only, no training")
        return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    args.dataset_size = 3168498  # hardcode train dataset size. Needed to compute number of steps for the lr scheduler
    # valid size : 32332 // test size : 32332
    print(args)
    model = LongformerKobart(args)
    dm = LongformerSummaryModule(args)

    # args.default_root_dir : logs
    wandb_logger = WandbLogger(project='longformer_kobart',name=f'Longformer_testing') # 이거는 당사자 편한대로 추가하셔도 되고, 안하셔도 됩니다~
    tb_logger = TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))

    checkpoint_callback = ModelCheckpoint(dirpath=args.default_root_dir, filename='model_chp/{epoch:02d}-{val_loss:.3f}', every_n_train_steps=args.checkpoint_interval, save_top_k=-1, verbose=True, monitor='val_loss', mode='min', save_last=True)

    trainer = pl.Trainer(gpus=args.gpus, 
                        accelerator='ddp' if torch.cuda.is_available() else None,
                        accumulate_grad_batches = 1,  # if 4, 4 batch > 16 batch, 이거 숫자 늘리면 learning_rate scheduler parameter 조정 필요
                        max_epochs=args.max_epochs,
                        check_val_every_n_epoch=0.1, # validation 몇번마다 돌릴것인지? > 0.1 epoch에 1번
                        logger=[wandb_logger,tb_logger],
                        callbacks=checkpoint_callback,
                        gradient_clip_val=0.0, # No gradiet clipping
                        log_every_n_steps=100) # logging frequency in training step,
    if not args.test:
        trainer.fit(model,dm)
    trainer.test(model)

if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="longformer_kobart")
    parser = LongformerKobart.add_model_specific_args(main_arg_parser)
    args = parser.parse_args()
    main(args)