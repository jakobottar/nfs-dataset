"""
implements dataset building and loading for the nuclear forensics SEM image dataset

Jakob Johnson
09/19/2023
"""

import os
from glob import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from PIL import Image

from .utils import filter_dataframe, get_metadata, print_green


def make_csv_from_filenames(src_dir: str, dest: str = "dataset.csv", num_threads: int = 4):
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

    # save df as a csv
    df.to_csv(dest, index=False)


def build_nfs_datasets(dataset_configs: dict) -> None:
    """
    Build the NFS dataset.
    """

    out = []
    for config in dataset_configs:

        # load csv
        df = pd.read_csv(config["src_file"])

        # filter train/val datasets
        train_val = filter_dataframe(df, config["filters"])

        # re-label "label" column, convert to int codes
        if config["label"] == "Route":
            train_val["Route"] = train_val["StartingMaterial"] + train_val["Material"]

        # print(list(train_val[config["label"]].astype("category").cat.categories))
        # print(train_val[config["label"]].astype("category").cat.codes)
        # print(f"num classes{len(train_val[config["label"]].astype('category').cat.categories)}")
        train_val["LabelStr"] = train_val[config["label"]].astype("category")
        train_val["Label"] = train_val[config["label"]].astype("category").cat.codes

        # make train/val split
        train_val = train_val.sample(frac=1).reset_index(drop=True)
        val = train_val.iloc[: int(len(train_val) * config["val_split"])]
        train = train_val.iloc[int(len(train_val) * config["val_split"]) :]

        # generate dataset based on df

        config["train_dataset"] = NFSDataset(train)
        config["val_dataset"] = NFSDataset(val)

        out.append(config)

    return out


class NFSDataset(torch.utils.data.Dataset):
    """
    Nuclear Forensics dataset.

    Args:
        dataframe: pandas dataframe of filenames and labels
        transform (callable, optional): Optional transform to be applied to a sample
    """

    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __load_image(self, datarow):

        filename = datarow["FileName"]
        databar_height = datarow["Databar_Height"]

        image = np.array(Image.open(filename))

        # change dtype if necessary
        if image.dtype == np.uint16:
            image = image / 256
            image = image.astype(np.uint8)

        # convert to RGB if grayscale
        if len(image.shape) == 2:
            image = np.array([image, image, image])
        else:
            image = image.transpose((2, 0, 1))

        # cut off infobar
        image = image[:, :-databar_height, :]

        return image

    def __getitem__(self, key):
        data = self.df.iloc[key].to_dict()

        # load and preprocess image
        image = self.__load_image(data)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(data["Label"])
