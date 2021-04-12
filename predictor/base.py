import torch
import pytorch_lightning as pl
from torch import optim
from dataset import *
import numpy as np
from argparse import ArgumentParser


class Predictor(pl.LightningModule):
    def __init__(self, network, args, learning_rate=1e-3, batch_size=2048):
        super(Predictor, self).__init__()
        # self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.net = network(args)
        self.args = args
        # self.train_dataset = train_dataset
        # self.test_dataset = test_dataset

    def forward(self, *args):
        return self.net(*args)

    def configure_optimizers(self):
        return optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate, weight_decay=0)

    def training_step(self, batch, batch_idx):
        correct, loss, info = self(*batch)
        # print("here")
        self.log('train_loss', loss, on_step=True)
        self.log('train_acc', correct / self.hparams.batch_size, on_step=True, prog_bar=True)
        # if info is not None:
        #     for k, v in info:
        #         self.log('train' + k, v)
        # print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        correct, loss, info = self(*batch)

        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', correct / self.hparams.batch_size, on_step=True, prog_bar=True)
        # if info is not None:
        #     for k, v in info:
        #         self.log('test' + k, v)
        return loss

    def test_step(self, batch, batch_idx):
        correct, loss, info = self(*batch)

        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_acc', correct / self.hparams.batch_size, on_step=True, prog_bar=True)
        # if info is not None:
        #     for k, v in info:
        #         self.log('test' + k, v)
        return loss

    # def test_dataloader(self):
    #     dataset = self.args.dataset
    #     test_dataset = dataset(self.args.data_dir, split=(0.7, 1), nbits=8, seqlen=self.args.seqlen)
    #     test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=8,
    #                                               batch_size=self.hparams.batch_size, shuffle=True)
    #     return test_loader

    def train_dataloader(self):
        dataset = self.args.dataset
        train_dataset = dataset(self.args.data_dir, split=(0, 0.7), nbits=8, seqlen=self.args.seqlen)
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=8,
                                                   batch_size=self.hparams.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        dataset = self.args.dataset
        test_dataset = dataset(self.args.data_dir, split=(0.7, 1), nbits=8, seqlen=self.args.seqlen)
        test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=8,
                                                  batch_size=self.hparams.batch_size, shuffle=True)
        return test_loader

    # def training_step_end(self, outputs):
    #     print(outputs)
    #     total_correct = torch.stack([x['train_correct'] for x in outputs]).mean()
    #     acc = total_correct / self.hparams.batch_size
    #     self.log('train_acc', acc)
    #     return {'train_acc': acc}

    # def test_epoch_end(self, outputs):
    #     total_correct = torch.stack([x['test_correct'] for x in outputs]).mean()
    #     acc = total_correct / self.hparams.batch_size
    #     self.log('test_acc', acc)
    #     return {'test_acc': acc}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--batch_size', type=int, default=512)
        return parser


class BellPredictor(Predictor):
    def __init__(self, network, args, learning_rate=1e-3, batch_size=2048):
        super().__init__(network, args, learning_rate, batch_size)

    def validation_step(self, batch, batch_idx):
        correct, loss, info = self(*batch)

        self.log('val_loss', loss, on_step=True, on_epoch=True)
        val_distribution = np.zeros((8, 2))
        for xybit, correct, total in info:
            val_distribution[xybit, :] += np.array([correct, total])
        # self.val_distribution += info
        return val_distribution

    def validation_epoch_end(self,outputs) -> None:
        val_distributions = outputs
        total_val_distribution = sum(val_distributions)
        self.log("val_mean_acc", np.sum(total_val_distribution[:, 0]) / np.sum(total_val_distribution[:, 1]),
                 prog_bar=True)
        self.log("val_max_acc", np.max(total_val_distribution[:, 0]) / np.max(total_val_distribution[:, 1]))
        self.log("val_min_acc", np.min(total_val_distribution[:, 0]) / np.min(total_val_distribution[:, 1]))
        print(total_val_distribution)
