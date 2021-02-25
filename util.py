import argparse
import pytorch_lightning as pl
from network.warpper import Warpper
from network.rcnn import RCNN
from network.self_attn import SelfAttention
from network.lstm_attn import AttentionModel
from network.fc import ResFC
from network.mylinformer import MyLinFormer
import time


def get_args(lit_model):
    parser = argparse.ArgumentParser(description='QRNG Argparser')
    parser.add_argument("--predict", type=int, default=0, dest="predict")
    parser.add_argument("--seed", type=int, default=int(time.time()), dest="seed")
    parser.add_argument("--epochs", type=int, default=40, dest="epochs")
    parser.add_argument("--seqlen", type=int, default=100, dest="seqlen")
    parser.add_argument("--model", type=str, default="FC", dest="model")
    parser.add_argument("--num_class", type=int, default=256, dest="num_class")
    parser.add_argument("--log_interval", type=int, default=20, dest="log_interval")
    parser.add_argument("--default_save_path", type=str, default="training", dest="default_save_path")
    parser.add_argument("--data_dir", type=str, default="/data1/hh/qrng_new/TEST-DATA-20200706/QRNG/40G/FINALDATA/",
                        dest="data_dir")

    parser = pl.Trainer.add_argparse_args(parser)
    parser = lit_model.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


def get_network(args):
    name = args.model
    network_dict = {
        "FC": lambda: ResFC(num_classes=args.num_class, input_bits=8, seqlen=args.seqlen),
        "Linformer": lambda: MyLinFormer(num_classes=args.num_class, num_tokens=256, dim=32, seq_len=args.seqlen,
                                         depth=5, heads=8,
                                         dim_head=16, k=32, one_kv_head=True, share_kv=False, reversible=True),
        "RCNN": lambda: Warpper(RCNN(batch_size=args.batch_size)),
        "SelfAtten": lambda: Warpper(SelfAttention(batch_size=args.batch_size)),
        "Atten": lambda: Warpper(AttentionModel(batch_size=args.batch_size))
    }
    return network_dict.get(name, None)
