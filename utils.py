import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn

IMAGE_SHAPE = (64, 64)

transform = transforms.Compose([
    transforms.Resize(IMAGE_SHAPE), # Resize the input image
    transforms.ToTensor(), # Convert to torch tensor (scales data into [0,1])
    transforms.Lambda(lambda t: (t * 2) - 1), # Scale data between [-1, 1]
])


reverse_transform = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2), # Scale data between [0,1]
    transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    transforms.Lambda(lambda t: t * 255.), # Scale data between [0.,255.]
    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)), # Convert into an uint8 numpy array
    transforms.ToPILImage(), # Convert to PIL image
])

def plot_noise_distribution(noise, predicted_noise):
    plt.hist(noise.cpu().numpy().flatten(), density = True, alpha = 0.8, label = "ground truth noise")
    plt.hist(predicted_noise.cpu().numpy().flatten(), density = True, alpha = 0.8, label = "predicted noise")
    plt.legend()
    plt.savefig('s.png')

def plot_noise_prediction(noise, predicted_noise):
    plt.figure(figsize=(15,15))
    f, ax = plt.subplots(1, 2, figsize = (5,5))
    ax[0].imshow(reverse_transform(noise))
    ax[0].set_title(f"ground truth noise", fontsize = 10)
    ax[1].imshow(reverse_transform(predicted_noise))
    ax[1].set_title(f"predicted noise", fontsize = 10)
    plt.savefig('a.png')

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        """
        dim = 10
        """
        self.dim = dim

    def forward(self, time):
        """ time=
        tensor([0,1,2,3])
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        """ embeddings=
        tensor([[    0.0000,     0.0000,     0.0000,     0.0000,     0.0000],
                [    1.0000,     0.1000,     0.0100,     0.0010,     0.0001],
                [    2.0000,     0.2000,     0.0200,     0.0020,     0.0002],
                [    3.0000,     0.3000,     0.0300,     0.0030,     0.0003]])
        """
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        """ embeddings=
        tensor([[  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
                [  0.8415,  0.0998,  0.0100,  0.0010,  0.0001,  0.5403,  0.9950,  0.9999,  1.0000,  1.0000],
                [  0.9093,  0.1987,  0.0200,  0.0020,  0.0002, -0.4161,  0.9801,  0.9998,  1.0000,  1.0000],
                [  0.1411,  0.2955,  0.0300,  0.0030,  0.0003, -0.9900,  0.9553,  0.9996,  1.0000,  1.0000]])
        """
        return embeddings

class Block(nn.Module):
    def __init__(self, channels_in, channels_out, time_embedding_dims, labels, num_filters = 3, downsample=True):
        super().__init__()

        self.time_embedding_dims = time_embedding_dims
        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dims)
        self.labels = labels
        if labels:
            self.label_mlp = nn.Linear(1, channels_out)

        self.downsample = downsample

        if downsample:
            self.conv1 = nn.Conv2d(channels_in, channels_out, num_filters, padding=1)
            self.final = nn.Conv2d(channels_out, channels_out, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(2 * channels_in, channels_out, num_filters, padding=1)
            self.final = nn.ConvTranspose2d(channels_out, channels_out, 4, 2, 1)

        self.bnorm1 = nn.BatchNorm2d(channels_out)
        self.bnorm2 = nn.BatchNorm2d(channels_out)

        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.time_mlp = nn.Linear(time_embedding_dims, channels_out)
        self.relu = nn.ReLU()

    def forward(self, x, t, **kwargs):
        o = self.bnorm1(self.relu(self.conv1(x)))
        embedded_t = self.time_embedding(t)
        o_time = self.relu(self.time_mlp(embedded_t))
        o = o + o_time[(..., ) + (None, ) * 2]

        if self.labels:
            label = kwargs.get('labels')
            o_label = self.relu(self.label_mlp(label))
            o = o + o_label[(..., ) + (None, ) * 2]

        o = self.bnorm2(self.relu(self.conv2(o)))

        return self.final(o)