from rouge import Rouge
import os, sys, argparse, logging

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from torch.optim import lr_scheduler
from model_data_loader import *
from model.make_longformer import *

from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

"""
todo
1. rouge 코드 짜기 (어떤 지표를 쓸지 안정해졌음 + batch_size 크기로 나옴)
"""

# model, tokenizer = load_YoYak()

# dataset  = FineTuningDataset(tokenizer = tokenizer, data_path = " ")
# data_loader = dataloader(dataset, batch_size = 4, collate_fn = collate_fn)

logging.getLogger("lightning").setLevel(logging.CRITICAL)

class YoYakFinetuningDataModule(pl.LightningDataModule):
    def __init__(self, args : dict ):
        super().__init__()
        self.batch_size = args.batch_size
        self.train_file_path = args.train_file
        self.valid_file_path = args.valid_file
        self.test_file_path = args.test_file
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
        # self.tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)

    def train_dataloader(self):
        train_dataset = FineTuningDataset(self.tokenizer, self.train_file_path, encoder_max_len = args.max_input_len, decoder_max_len = args.max_output_len)
        train = DataLoader(train_dataset, batch_size = self.batch_size, collate_fn = collate_fn)
        return train
    
    def val_dataloader(self):
        validation_dataset = FineTuningDataset(self.tokenizer, self.valid_file_path, encoder_max_len = args.max_input_len, decoder_max_len = args.max_output_len)
        validation = DataLoader(validation_dataset, batch_size = self.batch_size, collate_fn = collate_fn)
        return validation
    
    def test_dataloader(self):
        test_dataset = FineTuningDataset(self.tokenizer, self.test_file_path, encoder_max_len = args.max_input_len, decoder_max_len = args.max_output_len)
        test = DataLoader(test_dataset, batch_size = self.batch_size, collate_fn = collate_fn)
        return test

class YoYakFinetuningModule(pl.LightningModule):
    def __init__(self, hparams) :
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1')
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
        # self.model = LongformerBartForConditionalGeneration.from_pretrained(self.hparams.model.path)
        # self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.hparams.tokenizer)
        self.rouge = Rouge()

    def forward(self, inputs) : 
        encoder_inputs, decoder_inputs, labels, labels_str = inputs
        return self.model(input_ids=encoder_inputs['input_ids'],
                          attention_mask=encoder_inputs['attention_mask'],
                          decoder_input_ids=decoder_inputs['input_ids'],
                          decoder_attention_mask=decoder_inputs['attention_mask'],
                          labels=labels['input_ids'])

    def training_step(self, batch, batch_idx) :
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True, batch_size = self.hparams.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx) : 
        outs = self(batch)
        loss = outs.loss
        self.log("val_loss",loss, prog_bar=True, batch_size = self.hparams.batch_size)
        return loss
        

    def test_step(self, batch, batch_idx) :
        _, _, _, label_str = batch
        outs = self(batch)
        loss = outs.loss
        softmax = torch.nn.Softmax(outs)
        outs_str = self.tokenizer.decode(softmax)
        rouge_scores = self.rouge.get_scores(outs_str, label_str)
        #### rouge 함수 output이 scalar가 아님 rouge 1, 2, l 중에 무엇을 쓰고, recall, precision, f1 중에 뭘쓸까?
        rouge_1 = np.mean((score["rouge-1"]["f"] for score in rouge_scores))
        rouge_2 = np.mean((score["rouge-2"]["f"] for score in rouge_scores))
        rouge_L = np.mean((score["rouge-l"]["f"] for score in rouge_scores))
        metrics = {'test loss': loss, 'rouge 1': rouge_1, 'rouge 2' : rouge_2, 'rouge L' : rouge_L}
        self.log_dict(metrics, batch_size = self.hparams.batch_size)

    def configure_optimizers(self) :
        optimizer = AdamW(self.model.parameters(), lr = self.hparams.lr, correct_bias = True)
        num_workers = self.hparams.num_workers
        data_len = self.hparams.dataset_size
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps = num_warmup_steps, 
            num_training_steps = num_train_steps)
        
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        logging.info(f'num_train_steps : {num_train_steps}')
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        
        lr_scheduler = {"scheduler" : scheduler,
        "monitor" : "loss",
        "interval" : "step",
        "frequency" : 1}

        return [optimizer], [lr_scheduler]


    @staticmethod
    def add_model_specific_args(parser):
        # parser.add_argument('--train_file', type=str, default='./finetune_data/train.csv', help='train file path') 
        # parser.add_argument('--valid_file', type=str, default='./finetune_data/valid.csv', help='valid file path')
        # parser.add_argument('--test_file', type=str, default='./finetune_data/test.csv', help='test file path') 
        parser.add_argument('--train_file', type=str, default='./finetune_data/train.csv', help='train file path') 
        parser.add_argument('--valid_file', type=str, default='./finetune_data/test.csv', help='valid file path')
        parser.add_argument('--test_file', type=str, default='./finetune_data/test.csv', help='test file path') 
        
        parser.add_argument("--model_path", type=str, default='./model/longformer_kobart_initial_ckpt', help="Path to the checkpoint directory or model name")
        parser.add_argument("--tokenizer_path", type=str, default='./model/longformer_kobart_initial_ckpt')
        parser.add_argument("--default_root_dir", type=str, default='logs', help="parent directory of log files")
        
        parser.add_argument("--gpus", type=int, default=1, help="Number of gpus. 0 for CPU")
        parser.add_argument("--num_workers", type=int, default=6, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        
        parser.add_argument("--batch_size", type=int, default=4, help="Batch size")   
        parser.add_argument("--max_epochs", type=int, default=5, help="Number of epochs") 
        
        parser.add_argument('--lr',type=float,default=0.00003,help='The initial learning rate')
        parser.add_argument("--warmup_ratio", type=int, default=0.1, help="warmup ratio")
        
        parser.add_argument("--max_output_len", type=int, default=1024, help="maximum num of output length. Used for training and testing")
        parser.add_argument("--max_input_len", type=int, default=4096, help="maximum num of input length. Used for training and testing")
        
        parser.add_argument("--checkpoint_interval", type=int, default=5000, help="Number of training steps between checkpoints") # 우선 데이터셋의 1/10으로 해놓음

        parser.add_argument("--test", action='store_true', help="Test only, no training")
        return parser

def main(args) : 
    pl.seed_everything(args.seed)
    args.dataset_size = 89238
    print("\ncurrent working directory :", os.getcwd(), "\n")
    print(args)
    
    model = YoYakFinetuningModule(args)
    dm = YoYakFinetuningDataModule(args)

    wandb_logger = WandbLogger(
        project = "YoYak_finetuned", 
        name = "YoYak_finetuned on our dataset")
    
    tb_logger = TensorBoardLogger(os.path.join(args.default_root_dir, "tb_logs"))

    checkpoint_callback = ModelCheckpoint(dirpath=args.default_root_dir, 
    filename='model_chp/{epoch:02d}-{val_loss:.3f}', 
    every_n_train_steps=args.checkpoint_interval, 
    save_top_k=-1, 
    verbose=True, 
    monitor='val_loss', 
    mode='min', save_last=True)

    trainer = pl.Trainer(gpus = args.gpus, 
                    accumulate_grad_batches = 1,  # if 4, 4 batch > 16 batch, 이거 숫자 늘리면 learning_rate scheduler parameter 조정 필요
                    max_epochs=args.max_epochs,
                    val_check_interval=50, # validation 몇번마다 돌릴것인지? > 0.1 epoch에 1번
                    logger=[wandb_logger,tb_logger],
                    callbacks=checkpoint_callback,
                    gradient_clip_val=10.0, 
                    log_every_n_steps=10) # logging frequency in training step

    if not args.test:
        trainer.fit(model,dm)
    trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="YoYak_finetuned")
    parser = YoYakFinetuningModule.add_model_specific_args(main_arg_parser)
    args = parser.parse_args()
    main(args)