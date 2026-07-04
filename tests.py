# from model import SiameseRIZZNet
from TripletLoss.model import SiameseRIZZNet
# import torch

# model = SiameseRIZZNet()


# input_ = torch.rand((32, 1, 28, 28))

# output = model(input_)

# print(output.shape)


# sum_params = lambda M: sum(p.numel() for p in M.parameters()) 

# # parameters = sum([sum_params(M) for M in [Encoder, Decoder, Merger, Refiner]])
# parameters = sum_params(model)
# print("params: ", parameters)

# size_in_mb = parameters>>18 


# print("Model size: ", size_in_mb)


# from dataset import MNISTDataset
# import matplotlib.pyplot as plt

# data = MNISTDataset("./data/mnist_train.csv")

# anc, pos, neg = data[0]

# print(anc.shape)
# print(pos.shape)
# print(neg.shape)


# # plt.imshow(anc.reshape(28, 28), cmap='gray')
# # plt.show()
# # plt.imshow(pos.reshape(28, 28), cmap='gray')
# # plt.show()
# # plt.imshow(neg.reshape(28, 28), cmap='gray')
# # plt.show()

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import visualtorch
from torch import nn


model = SiameseRIZZNet(128, 0.01)
model.eval()
input_shape = input_shape = (1, 1, 28, 28)
color_map: dict = defaultdict(dict)
color_map[nn.Conv2d]["fill"] = "#E69F00"
color_map[nn.BatchNorm2d]["fill"] = "#009E73"
color_map[nn.ReLU]["fill"] = "#56B4E9"

img_default = visualtorch.render(model, input_shape, style="flow", color_map=color_map, legend=True)


dpi = 150  # rendered at 2x this in the final doc build (savefig.dpi=300 in conf.py)
plt.figure(figsize=(img_default.width / dpi, img_default.height / dpi), dpi=dpi)
plt.imshow(img_default)
plt.axis("off")
plt.tight_layout()
plt.show()