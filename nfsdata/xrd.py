import glob
import json
import os
from random import shuffle

import pandas as pd
from tqdm import tqdm

VAL_PERC = 0.2

import io
import os
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

COLS = "?,??,TwoTheta,Theta,Intensity"
## interpolation constants
X_MIN = 10
X_MAX = 70
NUM_POINTS = 4096


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
    data = data[(data["TwoTheta"] >= 10) & (data["TwoTheta"] <= 70)]

    ## interpolate the data to match a standard range of 10 to 90 degrees
    x = np.linspace(X_MIN, X_MAX, NUM_POINTS)  # delta of 0.1953125 degrees
    y = np.interp(x, data["TwoTheta"], data["Intensity"])

    # copy data to new dataframe
    data = pd.DataFrame({"TwoTheta": x, "Intensity": y})

    ## normalize the intensity by integrating the area under the curve
    data["Intensity"] = data["Intensity"] / data["Intensity"].sum()

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

    for _, sample in tqdm(dataset.iterrows(), dynamic_ncols=True, total=len(dataset)):
        ### process XRD file
        xrd_file = sample["xrd_file"]

        if xrd_file.endswith(".brml"):
            xrd_data = read_brml(os.path.join(xrd_src_dir, xrd_file))
        elif xrd_file.endswith(".txt") or xrd_file.endswith(".csv"):
            xrd_data = read_txt(os.path.join(xrd_src_dir, xrd_file))
        xrd_data = process_xrd_data(xrd_data)

        # drop "twotheta" column and convert to numpy
        xrd_data = xrd_data.drop(columns=["TwoTheta"])
        xrd_data = xrd_data.to_numpy().transpose(1, 0).astype(np.float32)

        # save data to disk
        # np.save("xrd.npy", xrd_data)
        # print(np.load("xrd.npy"))

        ### process SEM image
        # TODO: process SEM image


def make_paired_dataset(root_dir: str, dest_dir: str) -> None:

    # combine all json files in root_dir into one json file
    paired_data = []
    for fn in glob.glob(f"{root_dir}/*.json"):
        with open(fn, "r") as f:
            paired_data.extend(json.loads(f.read()))

    # split into train and val sets
    shuffle(paired_data)
    val_data = paired_data[: int(len(paired_data) * VAL_PERC)]
    train_data = paired_data[int(len(paired_data) * VAL_PERC) :]

    val_data = make_paired_dataframe(val_data)
    train_data = make_paired_dataframe(train_data)

    print(val_data)
    print(train_data)

    # process datasets
    print("Processing datasets...")
    process_paired_dataset(val_data, "../magni/data", os.path.join(dest_dir, "val"))
    process_paired_dataset(train_data, "../magni/data", os.path.join(dest_dir, "train"))


if __name__ == "__main__":
    make_paired_dataset("./xrd_datasets", "data")
