import torch
import torch.nn as nn

class CAE(nn.Module):
    """卷积自编码器"""
    def __init__(self, input_shape, filters):
        super(CAE, self).__init__()
        self.input_shape = input_shape
        self.filters = filters
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], filters[0], 5, stride=2, padding=2),  # 确保padding正确
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(),
            nn.Conv2d(filters[2], filters[3], 3, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(filters[3], filters[2], 3, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.ConvTranspose2d(filters[0], input_shape[0], 5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
