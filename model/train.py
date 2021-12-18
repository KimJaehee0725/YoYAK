import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from make_longformer import *
from transformers import PreTrainedTokenizerFast

# Dataset이 넘겨줘야할 것들
# encoder_inputs : token_ids, attention_mask
# decoder_inputs : token_ids, attention_mask
# label : token_ids 

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
    def __init__(self, params):
        super().__init__()
        self.args = params
        self.model = LongformerBartForConditionalGeneration.from_pretrained(args.model_path)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.args.tokenizer_path)
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
    
    # TO DO !
    # Add rouge score in test step
    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    # 수렴 이상하면 learning rate scheduler 추가하기
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, correct_bias=False)
        return optimizer

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        
	parser.add_argument('--train_file', type=str, default='/ /train.csv', help='train file')
        parser.add_argument('--valid_file', type=str, default='/ /valid.csv', help='valid file')
        parser.add_argument('--test_file', type=str, default='/ /test.csv', help='test file')
        
        parser.add_argument("--model_path", type=str, default='longformer_kobart', help="Path to the checkpoint directory or model name")
        parser.add_argument("--tokenizer_path", type=str, default='longformer_kobart')
        
        parser.add_argument("--gpus", type=int, default=1, help="Number of gpus. 0 for CPU")
        parser.add_argument("--num_workers", type=int, default=5, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        
	parser.add_argument("--max_epochs", type=int, default=2, help="Number of epochs")
        parser.add_argument("--max_output_len", type=int, default=1024, help="maximum num of output length. Used for training and testing")
        parser.add_argument("--max_input_len", type=int, default=4096, help="maximum num of input length. Used for training and testing")
        
        parser.add_argument('--lr',type=float,default=1e-7,help='The initial learning rate')

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

    wandb_logger = WandbLogger(project='longformer',name=f'Longformer_testing')

    # Early Stopping 추가하기
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', filename='{epoch:02d}-{val_loss:.3f}', save_top_k=2, verbose=True, monitor='val_loss', mode='min')
    print(args)

    trainer = pl.Trainer(gpus=args.gpus, 
                        distributed_backend='ddp' if torch.cuda.is_available() else None,
                        accumulate_grad_batches = 1 # if 4, 4 batch > 16 batch
                        max_epochs=args.max_epochs,
                        val_check_interval= 0.25, # check check validation set 4 times during a training epoch
                        check_val_every_n_epoch=1, 
                        logger=wandb_logger,
                        checkpoint_callback=checkpoint_callback,
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