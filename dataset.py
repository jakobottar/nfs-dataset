"""
implements dataset building and loading for the nuclear forensics SEM image dataset

Jakob Johnson
09/19/2023
"""

import os
from glob import glob
from multiprocessing import Pool

from pandas import DataFrame
from torch.utils.data import Dataset

from processing import get_metadata
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
    print("Getting filenames... ", end="")
    glob_pattern = os.path.join(src_dir, "*")
    filenames = glob(glob_pattern)
    print_green("done.")

    # get image metadata and hashes
    print("Getting metadata (this may take a while)... ", end="")
    with Pool(num_threads) as pool:
        image_metadata = pool.map(get_metadata, filenames)
    print_green("done.")

    # remove duplicates
    print("Removing duplicates... ", end="")
    df = DataFrame().from_dict(image_metadata)
    df = df.drop_duplicates(subset="Hash", keep="first")
    print_green("done.")

    # unique values
    for col in [
        "Material",
        # "Magnification",
        # "Resolution",
        # "HFW",
        "StartingMaterial",
        # "CalcinationTemp",
        # "CalcinationTime",
        # "AgingTime",
        # "AgingTemp",
        # "AgingHumidity",
        # "AgingOxygen",
        # "Impurity",
        # "ImpurityConcentration",
        "Detector",
        # "Coating",
        # "Replicate",
        # "Particle",
        # "Image",
        # "Date",
        "DetectorMode",
    ]:
        print(col)
        print(df[col].unique())

    # TODO: filter datasets according dataset_config.yml

    # TODO: package datasets into lmdbs


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
    build_nfs_dataset(
        "/scratch_nvme/jakobj/all-morpho-images/", "./data/processed", num_threads=16
    )
