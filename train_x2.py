import torch
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import random

import unet

from img2imgDataset import Image2ImageDatasetX2


class X2Updater(pl.LightningModule):
    def __init__(self, ckpt_128):
        super().__init__()
        self.cnn = unet.UNET()
        self.cnn_128 = unet.UNET()

        # load checkpoint
        ckpt = torch.load(ckpt_128)
        state_dict = {}
        for key, value in ckpt['state_dict'].items():
            if key[:4] == 'cnn.':
                state_dict[key[4:]] = value
        self.cnn_128.load_state_dict(state_dict)

    def training_step(self, batch, batch_idx):
        x_in, x_in_2, t_out = batch

        x_out = self.cnn_128(x_in)
        x_out = x_out.detach()

        for j in range(x_in_2.shape[0]):
            for ch in range(3):
                # randomly use src color ch
                if random.random() < 0.8:
                    x_in_2[j, [1 + ch], :] = TF.resize(
                        x_out[j, [ch], :], (512, 512), interpolation=InterpolationMode.BICUBIC)

        x_out_2 = self.cnn(x_in_2)

        loss = F.l1_loss(x_out_2, t_out)

        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.cnn.parameters(), lr=0.0001, weight_decay=1e-5)
        return opt

def main(args):
    root = args.dataset

    dataset = Image2ImageDatasetX2(
        "dat/images_color_train.dat", root + "linex2/", root + "colorx2/", train=True)
    train_iter = torch.utils.data.DataLoader(dataset, args.batchsize, shuffle=True, num_workers=args.num_workers, persistent_workers=True)

    # Set up a trainer
    updater = X2Updater(args.ckpt_128)

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
    parser.add_argument('--ckpt_128', default='models/128.ckpt',
                        help='Checkpoint of CNN 128.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_top_k', type=int, default=3)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)