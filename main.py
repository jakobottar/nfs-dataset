import random

import matplotlib.pyplot as plt

import nfsdata

# make csv file
# nfsdata.make_csv_from_filenames("/usr/sci/projs/DeepLearning/Jakob_Dataset/box/all-morpho-images/")

# read config file
configs = nfsdata.parse_config_file("./dataset_config.yaml")

# make datasets
datasets = nfsdata.build_nfs_datasets(configs)
print(datasets)

for n, dataset in enumerate(datasets):

    print(f"train dataset: {len(dataset['train_dataset'])}")
    print(f"val dataset  : {len(dataset['val_dataset'])}")

    # # make a grid of sample images
    fig, axs = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            img = dataset["train_dataset"][random.randint(0, len(dataset["train_dataset"]))][0]
            img = img.transpose((1, 2, 0))
            axs[i, j].imshow(img)
            axs[i, j].axis("off")

    fig.savefig(f"sample_{n}.png")
