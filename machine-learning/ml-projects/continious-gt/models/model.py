import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.stage1_block = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
        )

        self.stage2_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(300*300, 1000),
            nn.Linear(1000, 300)
        )

    def forward(self,x):
        x = self.stage1_block(x)
        #print(x.shape)
        x = self.stage2_block(x)
        return x




    


