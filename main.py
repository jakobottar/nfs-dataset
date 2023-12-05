import random

import matplotlib.pyplot as plt

import nfsdata

nfsdata.build_nfs_dataset(
    src_dir="/scratch_nvme/jakobj/all-morpho-images/",
    dest_dir="/scratch_nvme/jakobj/nfs/nfs-something/",
    config_file="./dataset_config.yml",
    num_threads=16,
)

dataset = nfsdata.NFSDataset("train", "/scratch_nvme/jakobj/nfs/nfs-something/")

# make a grid of 4 sample images
fig, axs = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        axs[i, j].imshow(dataset[random.randint(0, len(dataset))][0].permute(1, 2, 0))
        axs[i, j].axis("off")

fig.savefig("sample.png")

print(dataset)
