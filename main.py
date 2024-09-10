import nfsdata

# make csv file
# nfsdata.make_csv_from_filenames("/usr/sci/projs/DeepLearning/Jakob_Dataset/box/all-morpho-images/")

# read config file
configs = nfsdata.parse_config_file("./dataset_config.yaml")

# make datasets
datasets = nfsdata.build_nfs_datasets(configs)
print(datasets)
