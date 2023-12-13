import random

import matplotlib.pyplot as plt

import nfsdata

# from torchvision import transforms


nfsdata.build_lmdb_datasets(
    src_dir="/scratch_nvme/jakobj/nfs/all-morpho-images/",
    dest_dir="/scratch_nvme/jakobj/nfs/lmdbs/full",
    config_file="./dataset_config.yml",
    num_threads=16,
)

# nfsdata.build_imagefolder_datasets(
#     src_dir="/scratch_nvme/jakobj/all-morpho-images/",
#     dest_dir="/usr/sci/scratch/jakobj/nfs/tiles/128",
#     config_file="./dataset_config.yml",
#     num_threads=16,
# )

dataset = nfsdata.NFSDataset(
    "train",
    "/scratch_nvme/jakobj/nfs/lmdbs/full",
)

# dataset = nfsdata.ImageFolderDataset(
#     "train",
#     "/usr/sci/scratch/jakobj/nfs/tiles/128",
#     transform=transforms.ToTensor(),
# )

# make a grid of sample images
fig, axs = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        img = dataset[random.randint(0, len(dataset))][0].permute(1, 2, 0)
        axs[i, j].imshow(img)
        axs[i, j].axis("off")

fig.savefig("sample.png")

print(dataset)
