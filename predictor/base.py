import torch
import pytorch_lightning as pl
from torch import optim
from dataset import *
from argparse import ArgumentParser


class Predictor(pl.LightningModule):
    def __init__(self, network, args, train_dataset, test_dataset,learning_rate=1e-3, batch_size=512):
        super(Predictor, self).__init__()
        # self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.net = network()
        self.args = args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def forward(self, x, y):
        return self.net(x, y)

    def configure_optimizers(self):
        return optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate, weight_decay=0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        correct, loss = self(x, y)
        # print("here")
        self.log('train_loss',loss,on_step=True)
        self.log('train_acc', correct/self.hparams.batch_size, on_step=True)
        # print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        correct, loss = self(x, y)

        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', correct/self.hparams.batch_size, on_epoch=True)
        return loss

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset(), num_workers=8,batch_size=self.hparams.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset(),  num_workers=8,batch_size=self.hparams.batch_size, shuffle=False)
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
