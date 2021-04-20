import pytorch_lightning as pl
from pytorch_lightning import Trainer
from predictor.base import Predictor
from dataset.belltest_dataset import BellTestDataset
from dataset.dataset import QRNGDataset
from util import get_args, get_network, get_predictor

trainer = Trainer(log_every_n_steps=10)

if __name__ == '__main__':
    args = get_args(Predictor)
    pl.seed_everything(args.seed)

    # args.dataset = QRNGDataset
    args.dataset = BellTestDataset

    network = get_network(args)
    predictor = get_predictor(args)
    model = predictor(network, args, batch_size=args.batch_size, learning_rate=args.learning_rate)

    # trainer = pl.Trainer.from_argparse_args(args)
    # trainer = pl.Trainer.from_argparse_args(args)
    trainer = Trainer(gpus=args.gpus, max_epochs=args.epochs, default_root_dir="./models/", distributed_backend='ddp')
    trainer.fit(model)
