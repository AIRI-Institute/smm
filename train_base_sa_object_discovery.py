import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import os
import sys

from datasets.clevrtex_dataset import CLEVRTEXDataModule
from datasets.shapestacks_dataset import ShapeStacksDataModule

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)


from modules.SA import SlotAttentionModel
from models.mixture_dec_object_discovery import SlotAttentionMethod
from models.utils import ImageLogCallback, set_random_seed

import tensorboard
import json
import argparse



datamodules = {

    'clevrtex': CLEVRTEXDataModule,
    'shapestacks': ShapeStacksDataModule,

}

monitors = {
    'iou': 'avg_IoU',
    'ari': 'avg_ARI_FG',
    'ap': 'avg_AP@05',
}

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='')
parser.add_argument('--data_root', default='')
parser.add_argument('--log_name', default='base_sa')
parser.add_argument('--log_path', default='../../results/')
parser.add_argument('--ckpt_path', default='ckpt.pt.tar')
parser.add_argument('--test_ckpt_path', default='.ckpt')

parser.add_argument('--evaluate', type=str, default='ari', help='ari or iou')
parser.add_argument('--monitor', type=str, default='avg_ARI_FG', help='avg_ARI_FG or avg_IoU')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--max_steps', type=int, default=250000)
parser.add_argument('--max_epochs', type=int, default=100000)
parser.add_argument('--num_sanity_val_steps', type=int, default=1)
parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
parser.add_argument('--n_samples', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=128, help='batch size per GPU, if use 2 GPUs, change batch size to 64')
parser.add_argument('--gpus', type=int, default=0)

parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--resolution', type=int, nargs='+', default=[128, 128])
parser.add_argument('--init_resolution', type=int, nargs='+', default=[8, 8])
parser.add_argument('--encoder_channels', type=int, nargs='+', default=[64, 64, 64, 64])
parser.add_argument('--decoder_channels', type=int, nargs='+', default=[64, 64, 64, 64])
parser.add_argument('--encoder_strides', type=int, nargs='+', default=[2, 1, 1, 1])
parser.add_argument('--decoder_strides', type=int, nargs='+', default=[2, 2, 2, 2])
parser.add_argument('--encoder_kernel_size', type=int, default=5)
parser.add_argument('--decoder_kernel_size', type=int, default=5)

parser.add_argument('--is_logger_enabled', default=False, action='store_true')
parser.add_argument('--load_from_ckpt', default=False, action='store_true')
parser.add_argument('--use_rescale', default=False, action='store_true')

parser.add_argument('--lr_sa', type=float, default=4e-4)
parser.add_argument('--warmup_steps', type=int, default=5000)
parser.add_argument('--decay_steps', type=int, default=50000)

parser.add_argument('--num_iter', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=2)
parser.add_argument('--init_size', type=int, default=64)
parser.add_argument('--slot_size', type=int, default=64)
parser.add_argument('--mlp_size', type=int, default=128)


def main(args):
    print(args)
    set_random_seed(args.seed)
    args.monitor = monitors[args.evaluate]
    datamodule = datamodules[args.dataset](args)
    model = SlotAttentionModel(args)
    method = SlotAttentionMethod(model=model, datamodule=datamodule, args=args)
    method.hparams = args

    if args.is_logger_enabled:
        logger = pl_loggers.TensorBoardLogger(args.log_path, name=args.log_name) 
        arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
        arg_str = '__'.join(arg_str_list)
        log_dir = os.path.join(args.log_path, args.log_name)
        print(log_dir)
        logger.experiment.add_text('hparams', arg_str)
        callbacks = [LearningRateMonitor("step"), ImageLogCallback(), ModelCheckpoint(monitor=args.monitor, save_top_k=3, save_last=True, mode='max')] 
    else:
        logger = False
        callbacks = []
    
    trainer = Trainer(
        resume_from_checkpoint=args.ckpt_path if args.load_from_ckpt else None,
        logger=logger,
        default_root_dir=args.log_path,
        accelerator="ddp" if args.gpus > 1 else None,
        num_sanity_val_steps=args.num_sanity_val_steps,
        gpus=args.gpus,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        log_every_n_steps=50,
        callbacks=callbacks,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.grad_clip,
        # val_check_interval=3000,
    )
    trainer.fit(method)

if __name__ == "__main__":
    args = parser.parse_args()
    paths = json.load(open('./path.json', 'r'))
    data_paths = paths['data_paths']
    args.log_path = paths['log_path']
    args.data_root = data_paths[args.dataset]
    args.log_path += args.dataset
    main(args)

    

