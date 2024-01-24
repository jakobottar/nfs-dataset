import random

import matplotlib.pyplot as plt

import nfsdata

nfsdata.build_lmdb_datasets(
    src_dir="/usr/sci/projs/DeepLearning/Jakob_Dataset/box/all-morpho-images/",
    dest_dir="/scratch/jakobj/nfs/lmdbs/256",
    config_file="./dataset_config.yml",
    num_threads=16,
)

dataset = nfsdata.NFSDataset(
    "train",
    "/scratch/jakobj/nfs/lmdbs/256",
)

# make a grid of sample images
fig, axs = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        img = dataset[random.randint(0, len(dataset))][0].permute(1, 2, 0)
        axs[i, j].imshow(img)
        axs[i, j].axis("off")

fig.savefig("sample.png")

print(dataset)
