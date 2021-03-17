import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding
import torch as th


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        3. For random numbers, we will not use segment embedding

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        _token = th.eye(embed_size)
        _token = th.cat([_token, -th.ones(1, embed_size)], dim=0)
        _token.require_grad = False
        self.register_buffer('_token', _token)

        self.position = PositionalEmbedding(d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self._token[sequence.long()] + self.position(sequence)
        return self.dropout(x)
