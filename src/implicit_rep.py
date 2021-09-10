import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def remake_image(coords, img, model):
    img = np.zeros_like(img)
    for index, coord in enumerate(coords):
        if index % 1000 == 0:
            print(index)
        img[tuple(coord.numpy())] = model.forward(coord).detach()
    return img


def test_remake_image(coords, img, values):
    re_values = values
    img = np.zeros_like(img)
    for index, coord in enumerate(coords):
        img[tuple(coord.numpy())] = re_values[index]
    return img


def prepare_batch(img):
    coords = list()
    values = list()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            coord = (i, j)
            coords.append(coord)
            values.append(img[coord])
    coords = torch.tensor(coords)
    values = torch.tensor(values).unsqueeze(-1)
    return coords, values


class SINLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SINLinear, self).__init__()
        self.affine = nn.Linear(in_features, out_features)
        new_weights = 2*(torch.rand_like(self.affine.weight.data)-0.5) * np.sqrt(6 / in_features / 30)
        self.affine.weight.data = new_weights
        self.factor = 1

    def forward(self, x):
        return self.affine(self.factor * x)


class SIREN(nn.Module):
    def __init__(self):
        super(SIREN, self).__init__()
        hidden_dim = 512
        self.affine1 = SINLinear(2, hidden_dim)
        self.affine1.affine.weight.data = 2*(torch.rand_like(self.affine1.affine.weight.data)-0.5)
        self.affine2 = SINLinear(hidden_dim, hidden_dim)
        self.affine3 = SINLinear(hidden_dim, hidden_dim)
        self.affine4 = SINLinear(hidden_dim, 1)

    def forward(self, x):
        y = x.float()/2
        factor = 1
        y = torch.sin(factor*self.affine1(y))
        y = torch.sin(factor*self.affine2(y))
        y = torch.sin(factor*self.affine3(y))
        y = torch.sin(factor*self.affine4(y))
        return y


siren = SIREN()

folder = '/Users/raymondbaranski/Pictures/'
image = np.asarray(Image.open(folder + 'DataTNG.jpg'), dtype=float)
image = np.sum(image, axis=2)
image = image[::2, ::2]
image = image / np.max(image)

# coords, values = prepare_batch(image)
# recon_img = test_remake_image(coords, image, values)

optimizer = torch.optim.Adam(siren.parameters(), lr=0.005)
mseloss = nn.MSELoss()
train_iters = 100
for i in range(train_iters):
    if i % 10 == 0:
        print(i)
    coords, values = prepare_batch(image)
    indices = np.random.choice(list(range(torch.numel(values))), 1000)
    output = siren(coords[indices])
    loss = mseloss(output, values.float()[indices])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

coords = list()
for i in range(-200, 300, 1):
    for j in range(-200, 300, 1):
        coords.append((i, j))
coords = torch.tensor(coords)
image = np.zeros((500, 500))
recon_img = remake_image(coords, image, siren)
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.imshow(recon_img)
plt.show()


