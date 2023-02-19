import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import lnet

from img2imgDataset import Image2ImageDataset


class LnetUpdater(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l = lnet.LNET()

    def training_step(self, batch, batch_idx):
        t, x = batch

        l = self.l((x - 128) / 128)
        loss = F.l1_loss(l, (t[:, :1] - 128) / 128)

        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.l.parameters(), lr=0.0001, weight_decay=1e-5)
        return opt

def main(args):
    root = args.dataset

    dataset = Image2ImageDataset(
        "dat/images_color_train.dat", root + "line/", root + "color/", train=True)
    train_iter = torch.utils.data.DataLoader(dataset, args.batchsize, shuffle=True, num_workers=args.num_workers, persistent_workers=True)

    # Set up a trainer
    updater = LnetUpdater()

    checkpoint_callback = ModelCheckpoint(monitor='loss', save_last=True, save_top_k=args.save_top_k)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(updater, train_iter)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--dataset', '-i', default='./images/',
                        help='Directory of image files.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_top_k', type=int, default=3)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)