import torch
import os

# Create directory for data
os.mkdir("rbc_data")

# Read data
data = torch.load("rbc_data.pt")

# Standardization
velocity_norm = torch.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
std = torch.std(velocity_norm)  # Standard deviation of velocity norm
avg = torch.mean(data, dim=(0, 2, 3), keepdim=True)  # Mean velocity vector
data = (data - avg) / std
data = data[:, :, ::4, ::4]  # Downsample by a factor of 4
print("Mean velocity vector:", avg)
print("Standard deviation of velocity norm:", std)

# Divide each rectangular snapshot into 7 subregions
# data_prep shape: num_subregions * time * channels * w * h
data_prep = torch.stack([data[:, :, :, k * 64:(k + 1) * 64] for k in range(7)])

# Use sliding windows to generate samples
for j in range(0, 1500):
    for i in range(7):  #7 subregions
        torch.save(torch.FloatTensor(data_prep[i, j: j + 100]).double().float(), "rbc_data/sample_" + str(j * 7 + i) + ".pt")
