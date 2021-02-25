# from torch.nn import modules
from torch import nn
import torch
from linformer import LinformerLM


class MyLinFormer(nn.Module):
    def __init__(self, num_classes=256, num_tokens=256, dim=128, seq_len=4096, depth=12, heads=8, dim_head=128, k=64,
                 one_kv_head=True,
                 share_kv=False, reversible=True):
        super(MyLinFormer, self).__init__()
        self.linformer = LinformerLM(num_tokens, dim, seq_len, depth, heads, dim_head, k, one_kv_head, share_kv,
                                     reversible)
        self.output_fc = nn.Linear(seq_len * dim * heads, num_classes)

    def forward(self, x, y):
        last_layer = self.linformer(x.long())
        last_layer = last_layer.reshape(x.shape[0], -1)
        logits = self.output_fc(last_layer)
        logits = nn.Softmax(dim=1)(logits)
        # print(logits.shape, y.shape)
        loss = nn.CrossEntropyLoss()(logits, y)
        _, predict = torch.max(logits, dim=1)
        correct = (predict == y).sum()

        return correct, loss
