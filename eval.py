import argparse

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from models import SetPredictor
from datasets import CLEVR
from utils import average_precision_clevr


def evaluate(model, dataloader, device, limit=2):
    thrs = [-1, 1, 0.5, 0.25, 0.125, 0.0625]
    model.to(device)
    model.eval()
    predictions = []
    targets = []
    print('Making predictions...')
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            img = batch['image'].to(device)
            targ = batch['target'].to(device)
            with torch.no_grad():
                pred = model(img)
            targets.append(targ)
            predictions.append(pred['prediction'])
            if i == limit:
                break
    print('Predictions are ready. Calculating metrics...')
    predictions = torch.cat(predictions, dim=0).reshape(-1, 10, 19).detach().cpu().numpy()
    targets = torch.cat(targets, dim=0).reshape(-1, 10, 19).detach().cpu().numpy()
    for thr in thrs:
        ap = average_precision_clevr(predictions, targets, distance_threshold=thr)
        print(f"AP({thr}): {ap:.4}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        "-b",
                        type=int,
                        default=64)
    parser.add_argument("--model_path",
                        "-p",
                        type=str,
                        default='./gmm_base_detach.pth')
    parser.add_argument("--device",
                        "-d",
                        type=str,
                        default='cpu')
    parser.add_argument("--limit_data",
                        "-l",
                        default=2)
    args = parser.parse_args()
    pl.seed_everything(39)

    device = torch.device(args.device)
    data = CLEVR(
        images_path='./CLEVR_v1.0/images/val', 
        scenes_path='./CLEVR_v1.0/scenes/CLEVR_val_scenes.json',
        max_objs=10
        )
    dataloader = DataLoader(data, batch_size=args.batch_size)
    model = SetPredictor(num_slots=10)
    model.load_state_dict(torch.load('e725.pth', map_location=device), strict=False)
    limit = args.limit_data
    evaluate(model, dataloader, device, limit)
    print('*** END ***')


if __name__ == '__main__':
    main()
