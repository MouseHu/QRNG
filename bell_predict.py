import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from dataset.belltest_dataset import BellTestDataset
from predictor.base import Predictor
from util import get_args, get_network
import numpy as np

trainer = Trainer(log_every_n_steps=10)
p_bar = 0.6
if __name__ == '__main__':
    args = get_args(Predictor)
    pl.seed_everything(args.seed)

    args.dataset = BellTestDataset

    network = get_network(args)

    model = Predictor(network, args, batch_size=args.batch_size, learning_rate=args.learning_rate)
    model = model.load_from_checkpoint('./models/lightning_logs/version_31/checkpoints/epoch=0-step=3250.ckpt')
    # model = model.load_from_checkpoint('./models/lightning_logs/version_0/checkpoints/epoch=54-step=178804.ckpt')
    # trainer = pl.Trainer.from_argparse_args(args)
    # trainer = pl.Trainer.from_argparse_args(args)
    trainer = Trainer(checkpoint_callback=False, gpus=args.gpus, max_epochs=args.epochs, default_root_dir="./models/")

    test_dataloader = model.val_dataloader()
    # test_dataloader = model.train_dataloader()
    device = torch.device("cuda")

    model.to(device)
    # model.eval().to(device)
    test_loss = 0
    total_correct = 0
    final_result = np.zeros((8, 2))

    print(len(test_dataloader.dataset))
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            data = (d.to(device) for d in data)
            sum_correct, loss, info = model(*data)
            distribution = info['distribution']
            maxps, predicts, correct_answers, xybits = info['max_prop'], info['prediction'], info['correct'], info[
                'xybit']
            # final_result[:, 1] += np.bincount(xybits)
            for maxp, predict, correct_answer, xybits in zip(maxps, predicts, correct_answers, xybits):
                if maxp >= p_bar:
                    if predict == correct_answer:
                        final_result[xybits, 0] += 1
                    final_result[xybits, 1] += 1
            # for xybit, correct, total in distribution:
            #     final_result[xybit, :] += np.array([correct, total])
            if i % 100 == 0:
                print(final_result)
                # print(sum_correct.sum().item())
                print(distribution)
            # loss.backward()
            total_correct += sum_correct.sum().item()
        test_loss /= len(test_dataloader.dataset)
        print('===== Test set loss: {:.4f} acc {:.6f} ====='.format(test_loss,
                                                                    total_correct / len(test_dataloader.dataset)))

    print(final_result)
    print(final_result[:, 0] / final_result[:, 1])
