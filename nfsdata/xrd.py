import glob
import json
import os
from random import shuffle

import pandas as pd
import torch
from tqdm import tqdm

VAL_PERC = 0.30

import io
import os
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from .utils import get_hash, random_crop

COLS = "?,??,TwoTheta,Theta,Intensity"
## interpolation constants
X_MIN = 10
X_MAX = 70
NUM_POINTS = 4096

IMAGE_SIZE = 256
NUM_CROPS = 10


def parse_xml_file(filename) -> pd.DataFrame:

    ## load XML file
    root = ET.parse(filename).getroot()

    data = [COLS]
    ## the data we want is located at the following path in the XML file:
    for datapoint in root.findall("./DataRoutes/DataRoute/Datum"):
        data.append(datapoint.text)

    ## group into dataframe (the data is already in CSV format, so we can just read it as CSV)
    data = pd.read_csv(io.StringIO("\n".join(data)))

    return data


def read_raw(filename) -> pd.DataFrame:

    print(
        "Not feasable, the file format is not publically available\n use 'jade pattern converter' instead: https://www.icdd.com/jade-pattern-converter/"
    )

    return None


def read_brml(filename, tmp_folder="./tmp") -> pd.DataFrame:

    ## make tmp folder
    os.makedirs(tmp_folder, exist_ok=True)

    ## unzip the file into tmp folder
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(tmp_folder)

    ## read the RawData0.xml file
    data = parse_xml_file(os.path.join(tmp_folder, "Experiment0", "RawData0.xml"))

    # drop dummy variable columns
    data = data.drop(columns=["?", "??"])

    ## clean up tmp folder
    os.system(f"rm -rf {tmp_folder}")

    ## return the data
    return data


def read_txt(filename) -> pd.DataFrame:

    ## read the first line of the text file:
    with open(filename, "r") as f:
        first_line = f.readline()

        ## there's a couple different txt file formats in here
        if ";RAW4.00" in first_line:
            data_start_line = [i for i, l in enumerate(f) if l.rstrip() == "[Data]"][0] + 3  # skip 3 lines past [Data]
            data = pd.read_csv(
                filename,
                skiprows=data_start_line,
                sep=",",
                names=["TwoTheta", "Intensity", "x"],
            )
            data = data.drop(columns=["x"])

        elif "[Measurement conditions]" in first_line:
            data_start_line = [i for i, l in enumerate(f) if l.rstrip() == "[Scan points]"][
                0
            ] + 3  # skip 3 lines past [Scan points]
            data = pd.read_csv(
                filename,
                skiprows=data_start_line,
                sep=",",
                names=["TwoTheta", "Intensity", "x"],
            )
            data = data.drop(columns=["x"])

        elif "[°2Th.]" in first_line:
            data = pd.read_csv(
                filename,
                sep="\t",
                skiprows=1,
                names=["TwoTheta", "Intensity", "x", "xx", "xxx", "xxxx", "xxxxx"],
            )
            data = data.drop(columns=["x", "xx", "xxx", "xxxx", "xxxxx"])

        elif "Angle" in first_line or "Commander Sample ID" in first_line:
            data = pd.read_csv(filename, sep=r"\s+", skiprows=1, names=["TwoTheta", "Intensity"])

        elif "PDF#" in first_line:
            data = pd.read_csv(filename, sep=r"\s+", skiprows=16)
            data = data.loc[:, ["2-Theta", "I(f)"]].rename(columns={"2-Theta": "TwoTheta", "I(f)": "Intensity"})

    return data


def process_xrd_data(data: pd.DataFrame) -> pd.DataFrame:

    ## subtract the minimum intensity from all intensities
    data["Intensity"] = data["Intensity"] - data["Intensity"].min()

    ## drop datapoints greather than 70 degrees or less than 10 degrees
    data = data[(data["TwoTheta"] >= X_MIN) & (data["TwoTheta"] <= X_MAX)]

    ## interpolate the data to match a standard range of 10 to 70 degrees
    x = np.linspace(X_MIN, X_MAX, NUM_POINTS)
    y = np.interp(x, data["TwoTheta"], data["Intensity"])

    # copy data to new dataframe
    data = pd.DataFrame({"TwoTheta": x, "Intensity": y})

    ## normalize the intensity by integrating the area under the curve
    auc = np.trapz(data["Intensity"], data["TwoTheta"])
    data["Intensity"] = data["Intensity"] / auc
    # print(np.trapz(data["Intensity"], data["TwoTheta"]))

    return data


def make_paired_dataframe(dataset_list: list) -> pd.DataFrame:
    """
    expand SEM file column, create dataframe with paired SEM and XRD files
    """

    data = []
    for sample in dataset_list:
        for sem_file in sample["sem_files"]:
            newsample = {
                "route": sample["route"],
                "finalmat": sample["finalmat"],
                "xrd_file": sample["xrd_file"],
                "sem_file": sem_file,
            }
            data.append(newsample)

    return pd.DataFrame(data)


def process_paired_dataset(dataset: pd.DataFrame, xrd_src_dir, dest_dir) -> pd.DataFrame:
    """
    process paired dataset

    preprocess and write XRD files to CSV
    preprocess and write SEM files to PNG

    make final dataframe with new locations for files
    """

    # make directories
    os.makedirs(os.path.join(dest_dir, "xrd"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "sem"), exist_ok=True)

    new_dataset = []
    for _, sample in tqdm(dataset.iterrows(), dynamic_ncols=True, total=len(dataset)):
        ### process XRD file
        xrd_file = sample["xrd_file"]

        # get hash of xrd file
        xrd_hash = get_hash(os.path.join(xrd_src_dir, xrd_file))
        processed_xrd_filename = f"xrd/{xrd_hash[:16]}-{sample['route']}.npy"

        # check if file is already processed
        if not os.path.exists(os.path.join(dest_dir, processed_xrd_filename)):

            if xrd_file.endswith(".brml"):
                xrd_data = read_brml(os.path.join(xrd_src_dir, xrd_file))
            elif xrd_file.endswith(".txt") or xrd_file.endswith(".csv"):
                xrd_data = read_txt(os.path.join(xrd_src_dir, xrd_file))
            xrd_data = process_xrd_data(xrd_data)

            # drop "twotheta" column and convert to numpy
            xrd_data = xrd_data.drop(columns=["TwoTheta"])
            xrd_data = xrd_data.to_numpy().transpose(1, 0).astype(np.float32)

            # save data to disk
            np.save(os.path.join(dest_dir, processed_xrd_filename), xrd_data)

        ### process SEM image
        sem_hash = get_hash(sample["sem_file"])

        # load image
        image = np.array(Image.open(sample["sem_file"]))

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
        image = image[:, :-60, :]

        # randomly get NxN crops
        for i in range(NUM_CROPS):
            cropped_image = random_crop(image, (IMAGE_SIZE, IMAGE_SIZE))
            processed_sem_filename = f"sem/{sem_hash[:16]}-{sample['route']}-{i}.png"
            Image.fromarray(cropped_image.transpose((1, 2, 0))).save(os.path.join(dest_dir, processed_sem_filename))

            # add to dataset
            new_dataset.append(
                {
                    "route": sample["route"],
                    "finalmat": sample["finalmat"],
                    "xrd_file": processed_xrd_filename,
                    "sem_file": processed_sem_filename,
                }
            )

    df = pd.DataFrame(new_dataset)
    df.to_csv(os.path.join(dest_dir, "metadata.csv"), index=False)
    return df


def make_paired_dataset(root_dir: str, dest_dir: str) -> None:

    # combine all json files in root_dir into one json file
    paired_data = []
    for fn in glob.glob(f"{root_dir}/*.json"):
        with open(fn, "r") as f:
            paired_data.extend(json.loads(f.read()))

    # split into train and val sets
    paired_data = make_paired_dataframe(paired_data)

    paired_data = paired_data.sample(frac=1).reset_index(drop=True)
    val_data = paired_data[: int(len(paired_data) * VAL_PERC)]
    train_data = paired_data[int(len(paired_data) * VAL_PERC) :]

    # get unique route values from data
    val_routes = val_data["route"].unique()
    train_routes = train_data["route"].unique()

    print(f"Val routes: {len(val_routes)}, Train routes: {len(train_routes)}")

    # val_data = make_paired_dataframe(val_data)
    # train_data = make_paired_dataframe(train_data)

    print(val_data)
    print(train_data)

    # make directories
    os.makedirs(os.path.join(dest_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "train"), exist_ok=True)

    # process datasets
    print("Processing datasets...")
    process_paired_dataset(val_data, root_dir, os.path.join(dest_dir, "val"))
    process_paired_dataset(train_data, root_dir, os.path.join(dest_dir, "train"))


ROUTES = [
    "U3O8AUC",
    "U3O8MDU",
    "U3O8SDU",
    "U3O8UO4",
    "UO2AUCd",
    "UO2AUCi",
    "UO2MDU",
    "UO2SDU",
    "UO2UO4",
    "UO3AUC",
    "UO3MDU",
    "UO3SDU",
    "UO3UO4",
]

FINALMATS = ["U3O8", "UO2", "UO3"]


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = "train", sem_transform=None, xrd_transform=None, mode="paired"):
        self.root = root
        self.split = split
        self.sem_transform = sem_transform
        self.xrd_transform = xrd_transform
        self.mode = mode  # can be 'paired', 'sem', 'xrd'

        # load dataset metadata file
        try:
            self.df = pd.read_csv(os.path.join(self.root, split, "metadata.csv"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Dataset {self.root} does not exist, make sure it has been built.") from exc

        if self.mode == "paired":
            # convert route to label
            self.df["label"] = self.df["route"].apply(ROUTES.index)
        elif self.mode == "sem":
            # convert route to label
            self.df["label"] = self.df["route"].apply(ROUTES.index)
        elif self.mode == "xrd":
            # convert route to label
            self.df["label"] = self.df["finalmat"].apply(FINALMATS.index)
            # drop duplicates of xrd_file column
            self.df = self.df.drop_duplicates(subset=["xrd_file"])
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        self.classes = FINALMATS if self.mode == "xrd" else ROUTES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]

        # get sample
        if self.mode == "paired" or self.mode == "sem":
            sem = Image.open(os.path.join(self.root, self.split, sample["sem_file"])).convert("RGB")
            if self.sem_transform:
                sem = self.sem_transform(sem)
        if self.mode == "paired" or self.mode == "xrd":
            xrd = np.load(os.path.join(self.root, self.split, sample["xrd_file"]))
            if self.xrd_transform:
                xrd = self.xrd_transform(xrd)

        # get label
        label = np.int64(sample["label"])

        # return data
        if self.mode == "xrd":
            return xrd, label
        elif self.mode == "sem":
            return sem, label
        else:  # paired mode
            return xrd, sem, label

    def __repr__(self):
        return f"PairedDataset: {self.split} split with {self.__len__()} samples"
