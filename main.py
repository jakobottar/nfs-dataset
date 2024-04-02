import random

import matplotlib.pyplot as plt

import nfsdata

nfsdata.build_imagefolder_datasets(
    config_file="./dataset_config.yml",
    num_threads=16,
)

dataset = nfsdata.ImageFolderDataset(
    "train",
    "./data/end-material/",
)

print(len(dataset))

# make a grid of sample images
fig, axs = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        img = dataset[random.randint(0, len(dataset))][0]
        axs[i, j].imshow(img)
        axs[i, j].axis("off")

fig.savefig("sample.png")

print(dataset)
