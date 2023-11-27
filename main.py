import nfsdata

nfsdata.build_nfs_dataset(
    src_dir="/scratch_nvme/jakobj/all-morpho-images/",
    dest_dir="./data/",
    config_file="./dataset_config.yml",
    num_threads=16,
)

dataset = nfsdata.NFSDataset("train", "./data/")
print(dataset)
