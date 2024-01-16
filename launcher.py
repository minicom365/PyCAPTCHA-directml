from model.model import captcha_model, model_conv, model_resnet
from data.datamodule import captcha_dm
import lightning.pytorch as pl
import torch.optim as optim
import torch
import os
from utils.config_util import configGetter
from utils.arg_parsers import train_arg_parser

cfg  = configGetter('SOLVER')
lr = cfg['LR']
batch_size = cfg['BATCH_SIZE']
epoch = cfg['EPOCH']

def main(arg):
    pl.seed_everything(42)
    m = model_resnet()
    model = captcha_model(
        model=m, lr=lr)
    dm = captcha_dm(batch_size=batch_size)

    tb_logger = pl.loggers.TensorBoardLogger(
        args.log_dir, name=args.exp_name, version=2, default_hp_metric=False)
        
    trainer = pl.Trainer(deterministic=True,
                         accelerator="dml",
                         precision=32,
                         logger=tb_logger,
                         fast_dev_run=False,
                         max_epochs=epoch,
                         log_every_n_steps=50,
                         sync_batchnorm=True
                         )
    trainer.fit(model, datamodule=dm)
    os.makedirs(args.save_path, exist_ok=True)
    trainer.save_checkpoint(os.path.join(args.save_path, 'model.pth'))
    
if __name__ == "__main__":
    args = train_arg_parser()
    main(args)

    
