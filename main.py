import random

import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import nfsdata

# nfsdata.build_imagefolder_datasets(
#     config_file="./dataset_config.yml",
#     num_threads=16,
# )

# dataset = nfsdata.ImageFolderDataset(
#     "train",
#     "./data/routes/",
# )

# print(len(dataset))

# # make a grid of sample images
# fig, axs = plt.subplots(4, 4)
# for i in range(4):
#     for j in range(4):
#         img = dataset[random.randint(0, len(dataset))][0]
#         axs[i, j].imshow(img)
#         axs[i, j].axis("off")

# fig.savefig("sample.png")

# print(dataset)

nfsdata.make_paired_dataset("/scratch_nvme/jakobj/nfs/xrd-routes", "data/paired-xrd-sem")
# dataset = nfsdata.PairedDataset("./data/paired-xrd-sem-2", "train", mode="sem", sem_transform=transforms.ToTensor())

# # dataset = nfsdata.ImageFolderDataset("val", "/scratch_nvme/jakobj/nfs/routes/", transform=transforms.ToTensor())

# print(dataset)

# loader = data.DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False)


# mean = 0.0
# std = 0.0
# nb_samples = 0.0
# for batch, _ in tqdm(loader):
#     batch_samples = batch.size(0)
#     batch = batch.view(batch_samples, batch.size(1), -1)
#     mean += batch.mean(2).sum(0)
#     std += batch.std(2).sum(0)
#     nb_samples += batch_samples

# mean /= nb_samples
# std /= nb_samples

# print(f"dataset stats: \n\t{mean}\n\t{std}")
