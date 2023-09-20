"""
implements dataset building and loading for the nuclear forensics SEM image dataset

Jakob Johnson
09/19/2023
"""

import os
from glob import glob
from multiprocessing import Pool

import lmdb
import pyxis as px
import torch
import yaml
from pandas import DataFrame

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
        dataset_configs = yaml.full_load(file)  # TODO: make safe
    print_green("done.")

    # filter train/val datasets
    print("Filtering train/val datasets... ", end="", flush=True)
    train_val = filter_dataframe(df, dataset_configs["train"]["filters"])
    train_val["Label"] = train_val[dataset_configs["train"]["label"]]
    # TODO: integer conversion of labels

    # make train/val split
    train_val = train_val.sample(frac=1).reset_index(drop=True)
    val = train_val.iloc[: int(len(train_val) * dataset_configs["train"]["val-split"])]
    train = train_val.iloc[int(len(train_val) * dataset_configs["train"]["val-split"]) :]

    datasets = [{"name": "train", "dataframe": train}, {"name": "val", "dataframe": val}]
    del dataset_configs["train"]
    print_green("done.")

    # filter other datasets
    print("Filtering other datasets... ", end="", flush=True)
    for dataset in dataset_configs:
        datasets.append(
            {
                "name": dataset,
                "dataframe": filter_dataframe(df, dataset_configs[dataset]["filters"]),
            }
        )
    print_green("done.")

    # package datasets into lmdbs
    for dataset in datasets:
        print(f"Building {dataset['name']} dataset... ", end="", flush=True)
        make_lmdb(dest_dir, dataset["name"], dataset["dataframe"])
        print_green("done.")

    # verify datasets
    for dataset in glob(os.path.join(dest_dir, "*")):
        with px.Reader(dirpath=dataset) as db:
            print(f"{dataset}:\n    number of samples: {len(db)}")
            sample = db.get_sample(17)
            for key in db.get_data_keys():
                print(f"    '{key}' <- dtype: {sample[key].dtype}, shape: {sample[key].shape}")

    print("datasets built!")


class NFSDataset(torch.utils.data.Dataset):
    """
    Nuclear Forensics dataset.

    Args:
        split (string): Dataset split to load, one of "train", "val", or ood splits defined in dataset_config.yml
        root_dir (string): Root directory of built dataset
        transform (callable, optional): Optional transform to be applied to a sample
    """

    def __init__(self, split: str, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dirpath = os.path.join(self.root_dir, split)

        # check if dataset exists
        try:
            self.db = px.Reader(self.dirpath, lock=False)
        except lmdb.Error as exc:
            raise FileNotFoundError(f"Dataset {self.dirpath} does not exist, make sure it has been built.") from exc

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, key):
        data = self.db[key]
        for k in data.keys():
            data[k] = torch.from_numpy(data[k])

        if self.transform:
            data["image"] = self.transform(data["image"].to(torch.float))

        return data["image"], data["label"]

    def __repr__(self):
        if self.transform:
            format_string = "Transforms: "
            for t in self.transform.transforms:
                format_string += "\n"
                format_string += f"        {t}"
            format_string += "\n"
            return str(self.db) + "\n" + format_string
        else:
            return str(self.db)


if __name__ == "__main__":
    # build_nfs_dataset("/usr/sci/scratch/jakobj/all-morpho-images/", "./data/processed", num_threads=16)
    build_nfs_dataset("/scratch_nvme/jakobj/all-morpho-images/", "./data/", num_threads=16)

    dataset = NFSDataset("train", "./data/")
    print(dataset)
