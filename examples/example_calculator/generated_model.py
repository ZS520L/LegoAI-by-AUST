import torch.nn as nn


class GeneratedModel(nn.Module):
    def __init__(self):
        super(GeneratedModel, self).__init__()
        self.layer_1 = nn.Conv2d(3, 1, 3, stride=16)
        self.layer_2 = nn.Flatten()
        self.layer_3 = nn.Linear(196, 100)
        self.layer_4 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        return x
