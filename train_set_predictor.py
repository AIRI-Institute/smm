import argparse

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from models import SetPredictor
from datasets import CLEVR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        "-b",
                        type=int,
                        default=512)
    parser.add_argument("--save_path",
                        "-s",
                        type=str,
                        default='./weights.pth')
    args = parser.parse_args()
    pl.seed_everything(39)

    train_data = CLEVR(
        images_path='./CLEVR_v1.0/images/train', 
        scenes_path='./CLEVR_v1.0/scenes/CLEVR_train_scenes.json',
        max_objs=10
    )
    val_data = CLEVR(
        images_path='./CLEVR_v1.0/images/val', 
        scenes_path='./CLEVR_v1.0/scenes/CLEVR_val_scenes.json',
        max_objs=10
    )
    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_data, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    model = SetPredictor(num_slots=10)
    trainer = pl.Trainer(
        gpus=[0],
        max_steps=150000,
        enable_progress_bar=True
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    torch.save(model.state_dict(), args.save_path)


if __name__ == '__main__':
    main()