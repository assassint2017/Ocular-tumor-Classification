"""
vgg-like网络脚本
"""

import torch.nn as nn

dropout_rate = 0.1


class VGG(nn.Module):
    """

    共294976可训练的参数,30W左右
    """
    def __init__(self, bias=True):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1, bias=bias),
            # nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(8, 8, 3, padding=1, bias=bias),
            # nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(3, 2, padding=1),
            nn.Dropout(dropout_rate)
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=bias),
            # nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, padding=1, bias=bias),
            # nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(3, 2, padding=1),
            nn.Dropout(dropout_rate)
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=bias),
            # nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1, bias=bias),
            # nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(3, 2, padding=1),
            nn.Dropout(dropout_rate)
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=bias),
            # nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1, bias=bias),
            # nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(3, 2, padding=1),
            nn.Dropout(dropout_rate)
        )

        self.stage5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=bias),
            # nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1, bias=bias),
            # nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AvgPool2d(14),
        )

        self.clf = nn.Linear(128, 2)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal(layer.weight.data, 0.25)
                nn.init.constant(layer.bias.data, 0)

    def forward(self, inputs):

        outputs = self.stage1(inputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.stage4(outputs)
        outputs = self.stage5(outputs)

        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.clf(outputs)

        return outputs

# # 计算网络参数数量
# num_parameter = .0
# net = VGG()
# for item in net.modules():
#
#     if isinstance(item, nn.Conv2d):
#         num_parameter += (item.weight.size(0) * item.weight.size(1) * item.weight.size(2) * item.weight.size(3))
#
#         if item.bias is not None:
#             num_parameter += item.bias.size(0)
#
#     elif isinstance(item, nn.PReLU):
#         num_parameter += item.num_parameters
#
#     elif isinstance(item, nn.BatchNorm2d):
#         num_parameter += item.num_features
#
#
# print(num_parameter)
