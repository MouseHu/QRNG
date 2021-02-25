from dataset.dataset import QRNGDataset
from predictor.base import Predictor
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from util import get_args, get_network

trainer = Trainer(log_every_n_steps=10)

if __name__ == '__main__':
    args = get_args(Predictor)
    pl.seed_everything(args.seed)

    train_dataset = lambda: QRNGDataset(args.data_dir, split=(0, 0.7), nbits=8, seqlen=args.seqlen)
    test_dataset = lambda: QRNGDataset(args.data_dir, split=(0.7, 1), nbits=8, seqlen=args.seqlen)

    network = get_network(args)

    model = Predictor(network, args, train_dataset, test_dataset)

    # trainer = pl.Trainer.from_argparse_args(args)
    # trainer = pl.Trainer.from_argparse_args(args)
    trainer = Trainer(checkpoint_callback=False, gpus=args.gpus, max_epochs=args.epochs)
    trainer.fit(model)
