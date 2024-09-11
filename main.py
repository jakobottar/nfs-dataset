from torch.utils.data import DataLoader

import nfsdata

# make csv file
# nfsdata.make_csv_from_filenames("/usr/sci/projs/DeepLearning/Jakob_Dataset/box/all-morpho-images/")

# read config file
configs = nfsdata.parse_config_file("./dataset_config.yaml")

# make datasets
datasets = nfsdata.build_nfs_datasets(configs)
print(datasets)

for dataset in datasets:
    train_dataloader = DataLoader(dataset["train_dataset"], batch_size=4)

    batch = next(iter(train_dataloader))
    print(batch)
