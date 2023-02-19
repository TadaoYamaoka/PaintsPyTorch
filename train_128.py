import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import unet
import lnet

from img2imgDataset import Image2ImageDataset


class GanUpdater(pl.LightningModule):
    def __init__(self, ckpt_lnet):
        super().__init__()
        self.cnn = unet.UNET()
        self.dis = unet.DIS()
        self.l = lnet.LNET()

        # load checkpoint
        ckpt = torch.load(ckpt_lnet)
        state_dict = {}
        for key, value in ckpt['state_dict'].items():
            if key[:2] == 'l.':
                state_dict[key[2:]] = value
        self.l.load_state_dict(state_dict)

        self.automatic_optimization = False

    def loss_cnn(self, x_out, t_out, y_out, lam1=1, lam2=1, lam3=10):
        loss_rec = lam1 * (F.l1_loss(x_out, t_out))
        loss_dis = F.cross_entropy(y_out, torch.zeros(y_out.shape[0], dtype=torch.int64, device=self.device))
        loss_adv = lam2 * loss_dis
        l_t = self.l((t_out - 128) / 128)
        l_x = self.l((x_out - 128) / 128)
        loss_l = lam3 * (F.l1_loss(l_x, l_t))
        loss = loss_rec + loss_adv + loss_l
        self.log_dict({ 'cnn/loss': loss, 'cnn/loss_rec': loss_rec, 'cnn/loss_adv': loss_adv, 'cnn/loss_l': loss_l }, prog_bar=True)
        return loss

    def loss_dis(self, y_real, y_fake):
        L1 = F.cross_entropy(y_real, torch.zeros(y_real.shape[0], dtype=torch.int64, device=self.device))
        L2 = F.cross_entropy(y_fake, torch.ones(y_fake.shape[0], dtype=torch.int64, device=self.device))
        loss = L1 + L2
        self.log_dict({ 'dis/loss': loss }, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        x_in, t_out = batch

        x_out = self.cnn(x_in)

        cnn_optimizer, dis_optimizer = self.optimizers(use_pl_optimizer=True)

        y_out = self.dis(x_out)

        loss_cnn = self.loss_cnn(x_out, t_out, y_out)
        cnn_optimizer.zero_grad()
        self.manual_backward(loss_cnn)
        cnn_optimizer.step()

        y_fake = self.dis(x_out.detach())
        y_real = self.dis(t_out)

        loss_dis = self.loss_dis(y_real, y_fake)
        dis_optimizer.zero_grad()
        self.manual_backward(loss_dis)
        dis_optimizer.step()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.cnn.parameters(), lr=0.0001, weight_decay=1e-5)
        opt_d = torch.optim.Adam(self.dis.parameters(), lr=0.0001, weight_decay=1e-5)
        return [opt, opt_d]

def main(args):
    root = args.dataset

    dataset = Image2ImageDataset(
        "dat/images_color_train.dat", root + "line/", root + "color/", train=True)
    train_iter = torch.utils.data.DataLoader(dataset, args.batchsize, shuffle=True, num_workers=args.num_workers, persistent_workers=True)

    # Set up a trainer
    updater = GanUpdater(args.ckpt_lnet)

    checkpoint_callback = ModelCheckpoint(monitor='cnn/loss', save_last=True, save_top_k=args.save_top_k)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(updater, train_iter)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--dataset', '-i', default='./images/',
                        help='Directory of image files.')
    parser.add_argument('--ckpt_lnet', default='models/lnet.ckpt',
                        help='Checkpoint of lnet.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_top_k', type=int, default=3)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)