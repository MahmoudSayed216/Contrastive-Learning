# from model import SiameseRIZZNet
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


from dataset import MNISTDataset
import matplotlib.pyplot as plt

data = MNISTDataset("./data/mnist_train.csv")

anc, pos, neg = data[0]

print(anc.shape)
print(pos.shape)
print(neg.shape)


# plt.imshow(anc.reshape(28, 28), cmap='gray')
# plt.show()
# plt.imshow(pos.reshape(28, 28), cmap='gray')
# plt.show()
# plt.imshow(neg.reshape(28, 28), cmap='gray')
# plt.show()