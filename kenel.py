import torch



d0 = (256 * 0.1 / 2) ** 2
m0 = (256 - 1) / 2.
x_grid = torch.arange(256)
kernel = 1 - torch.exp(-((x_grid - m0) ** 2.) / (2 * d0))
print(kernel)