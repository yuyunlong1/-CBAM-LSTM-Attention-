import torch
from torch import nn
import torch.nn.functional as F
import warnings
import math, copy

warnings.filterwarnings('ignore')


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        # assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        query, key, value = [l(x) for l, x in zip(self.linears, (
            query, key, value))]  # (batch_size, seq_length, d_model), use first 3 self.linears
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x in (query, key, value)]  # (batch_size, h, seq_length, d_k)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)


class CAM(nn.Module):
    def __init__(self, num_channels):
        super(CAM, self).__init__()
        self.num_channels = num_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.F1 = nn.Conv2d(self.num_channels, self.num_channels // 2, kernel_size=1, padding='same')
        self.F2 = nn.Conv2d(self.num_channels // 2, self.num_channels, kernel_size=1, padding='same')
        self.bn = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        z = self.avg_pool(x)
        z = self.relu(self.F1(z))
        z = self.bn(self.sigmoid(self.F2(z)))
        return x + torch.mul(x, z)


class EAM(nn.Module):
    def __init__(self, num_channels):
        super(EAM, self).__init__()
        self.num_channels = num_channels
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.F3 = nn.Conv2d(self.num_channels, 1, kernel_size=1)
        self.Fs = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, padding='same')
        self.bn = nn.BatchNorm2d(self.num_channels)

    def forward(self, x):
        s1 = F.sigmoid(self.F3(x))
        s2 = self.bn(F.relu(self.Fs(x)))
        s3 = torch.mul(s2, s1)
        return x + s3


class CBAM(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CBAM, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=2, stride=1, padding='same')
        self.ChannelAttention1 = CAM(self.output_channels)
        self.SpatialAttention1 = EAM(self.output_channels)
        self.pooling1 = nn.MaxPool2d((1, 2))
        self.bn1 = nn.BatchNorm2d(self.output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.ChannelAttention1(x)
        x = self.SpatialAttention1(x)
        return self.pooling1(x)


class Net(nn.Module):
    def __init__(self, input_channels, output_channels1, output_channels2, hidden_size):
        super(Net, self).__init__()
        self.input_channels = input_channels
        self.output_channels1 = output_channels1
        self.output_channels2 = output_channels2
        self.hidden_size = hidden_size
        self.CBAM1 = CBAM(self.input_channels, self.output_channels1)
        self.CBAM2 = CBAM(self.output_channels1, self.output_channels2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=self.hidden_size, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.attention = MultiHeadAttention(4, hidden_size * 2)
        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.fc2 = nn.Linear(self.hidden_size * 2, 1)
        self.con1d = nn.Conv1d(20, 1, 1, padding='same')

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.CBAM1(x)
        x = self.CBAM2(x)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.attention(x, x, x)
        x = self.fc1(x)
        x = self.con1d(x)
        x = x.squeeze(1)
        x = self.fc2(x)
        return x
