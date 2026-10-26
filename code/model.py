import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for width, kernel in [(16, 15), (32, 9), (64, 5)]: # 这里有三组卷积, 每经过一层卷积, 输出长度变为一半
            layers += [nn.Conv1d(channels, width, kernel, stride=2, padding=kernel//2),
                       nn.GroupNorm(4, width), # 分组归一化
                       nn.SiLU()] # SiLU (sigmoid + relu: x * sigmoid)
            channels = width
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return F.adaptive_avg_pool1d(self.layers(x), 8).flatten(1) # 每个channel的数据平均池化为8个特征值, 64个channel, 所以一共512个特征值, 所以输出就是[num-batch, 512]

class BPNet(nn.Module):
    def __init__(self, scg_channels=1, bcg_channels=1):
        super().__init__()
        self.scg, self.bcg = Encoder(scg_channels), Encoder(bcg_channels)
        self.head = nn.Sequential(nn.Linear(1024, 64), nn.SiLU(),
                                  nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, scg, bcg):
        # Inputs [batch, channels, samples]; output mmHg, [SBP, DBP].
        z = self.head(torch.cat([self.scg(scg), self.bcg(bcg)], dim=1))
        dbp = z[:, 0]
        pp = 40.0 * F.softplus(z[:, 1] + 0.54132485) # >>> math.log(math.e - 1) = 0.541324854612918 这里是为了让第二个模型输出为0的时候对应正常脉压: 40
        return torch.stack([dbp + pp, dbp], dim=1)

if __name__ == '__main__':
    model = BPNet()
    y = model(torch.randn(4, 1, 1000), torch.randn(4, 1, 1000))
    assert y.shape == (4, 2) and torch.all(y[:, 0] > y[:, 1])
    print('Shape:', tuple(y.shape))
    print('Parameters:', sum(p.numel() for p in model.parameters()))
