"""
implements dataset building and loading for the nuclear forensics SEM image dataset

Jakob Johnson
09/19/2023
"""

import os
from glob import glob
from multiprocessing import Pool

import yaml
from pandas import DataFrame
from torch.utils.data import Dataset

from processing import filter_dataframe, get_metadata, make_lmdb
from utils import print_green


def build_nfs_dataset(
    src_dir: str,
    dest_dir: str,
    num_threads: int = 4,
    config_file: str = "./dataset_config.yml",
) -> None:
    """
    Build the NFS dataset.

    Args:
        src_dir (string): source directory of image files
        dest_dir (string): destination directory for dataset
        num_threads (int): number of threads to use for processing
        config_file (string): path to dataset config file
    """

    # get filenames
    print("Getting filenames... ", end="", flush=True)
    glob_pattern = os.path.join(src_dir, "*")
    filenames = glob(glob_pattern)
    print_green("done.")

    # get image metadata and hashes
    print("Getting metadata (this may take a while)... ", end="", flush=True)
    with Pool(num_threads) as pool:
        image_metadata = pool.map(get_metadata, filenames)
    print_green("done.")

    # remove duplicates
    print("Removing duplicates... ", end="", flush=True)
    df = DataFrame().from_dict(image_metadata)
    df = df.drop_duplicates(subset="Hash", keep="first")
    print_green("done.")

    # unique values
    # for col in [
    #     "Material",
    #     # "Magnification",
    #     # "Resolution",
    #     "HFW",
    #     "StartingMaterial",
    #     # "CalcinationTemp",
    #     # "CalcinationTime",
    #     # "AgingTime",
    #     # "AgingTemp",
    #     # "AgingHumidity",
    #     # "AgingOxygen",
    #     # "Impurity",
    #     # "ImpurityConcentration",
    #     "Detector",
    #     "Coating",
    #     # "Replicate",
    #     # "Particle",
    #     # "Image",
    #     # "Date",
    #     "Detector",
    #     "DetectorMode",
    # ]:
    #     print(col)
    #     print(df[col].unique())

    # read config file
    print("Reading config file... ", end="", flush=True)
    with open(config_file, "r") as file:
        dataset_configs = yaml.full_load(file)  ## TODO: make safe
    print_green("done.")

    # filter train/test datasets
    print("Filtering train/test datasets... ", end="", flush=True)
    train_test = filter_dataframe(df, dataset_configs["train"]["filters"])
    train_test["Label"] = train_test[dataset_configs["train"]["label"]]
    # TODO: integer conversion of labels

    # make train/val split
    train_test = train_test.sample(frac=1).reset_index(drop=True)
    test = train_test.iloc[: int(len(train_test) * dataset_configs["train"]["test-split"])]
    train = train_test.iloc[int(len(train_test) * dataset_configs["train"]["test-split"]) :]

    datasets = [{"name": "train", "dataframe": train}, {"name": "test", "dataframe": test}]
    del dataset_configs["train"]
    print_green("done.")

    # filter other datasets
    print("Filtering other datasets... ", end="", flush=True)
    # TODO: filter other datasets
    print_green("done.")

    # package datasets into lmdbs
    for dataset in datasets:
        print(f"Building {dataset['name']} dataset... ", end="", flush=True)
        make_lmdb(dest_dir, dataset["name"], dataset["dataframe"])
        print_green("done.")

    # TODO: verify datasets


class NFSDataset(Dataset):
    """
    Nuclear Forensics dataset.

    Args:
        root_dir (string): Root directory of built dataset
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # TODO: check if dataset exists

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, idx):
        pass


if __name__ == "__main__":
    # build_nfs_dataset("/usr/sci/scratch/jakobj/all-morpho-images/", "./data/processed", num_threads=16)
    build_nfs_dataset("/scratch_nvme/jakobj/all-morpho-images/", "./data/", num_threads=16)
