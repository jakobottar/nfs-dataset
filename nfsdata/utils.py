import hashlib
import os
import re

import numpy as np
import pandas as pd
import pyxis as px
import tifffile
import yaml
from PIL import Image
from tqdm import tqdm

BUF_SIZE = 65536


def parse_config_file(filename: str) -> dict:
    """parse yaml config file"""
    with open(filename, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    new_config = []
    for x in config:
        tmp_config = {"name": x}
        tmp_config.update(config[x])
        new_config.append(tmp_config)

    return new_config


def random_crop(img, crop_size):
    """Randomly crop image to crop_size"""
    assert img.shape[1] >= crop_size[0]
    assert img.shape[2] >= crop_size[1]

    x = np.random.randint(0, img.shape[1] - crop_size[0] + 1)
    y = np.random.randint(0, img.shape[2] - crop_size[1] + 1)

    return img[:, x : x + crop_size[0], y : y + crop_size[1]]


def get_metadata(full_filename):
    """
    Break down filename and return dict of fields from the filename's info
    """
    metadata = {
        "Material": None,
        "Magnification": None,
        "Resolution": None,
        "HFW": None,
        "StartingMaterial": None,
        "CalcinationTemp": None,
        "CalcinationTime": None,
        "AgingTime": None,
        "AgingTemp": None,
        "AgingHumidity": None,
        "AgingOxygen": None,
        "Impurity": None,
        "ImpurityConcentration": None,
        "Detector": None,
        "Coating": None,
        "Replicate": None,
        "Particle": None,
        "Image": None,
        "Date": None,
    }

    # remove .tif extension

    name = os.path.basename(full_filename)[:-4]

    name_split = name.split(sep="_")

    # catch missing NA value
    if name_split[12] not in ["NA", "8wtpct"]:
        name_split.insert(12, "NA")

    # catch Alpha_UO3 and Am_UO3 naming mistakes
    if name_split[0].lower() == "am":
        del name_split[0]
        name_split[0] = "am" + name_split[0]

    if name_split[0].lower() == "alpha":
        del name_split[0]
        name_split[0] = "alpha" + name_split[0]

    for i, key in enumerate(metadata.keys()):
        try:
            metadata[key] = name_split[i]
        except IndexError:
            metadata[key] = "NA"

    ### Reformatting block:

    # format material
    metadata["Material"] = format_material(metadata["Material"])

    # format magnification
    metadata["Magnification"] = remove_units(metadata["Magnification"])

    # format resolution
    # TODO

    # format HFW, done in tif metadata section

    # format starting material
    metadata["StartingMaterial"] = format_starting_material(metadata["StartingMaterial"])

    # format calcination temp
    metadata["CalcinationTemp"] = remove_units(metadata["CalcinationTemp"])

    # format calcination time
    # TODO: convert to same unit
    # metadata["CalcinationTime"] = remove_units(metadata["CalcinationTime"])

    # format aging time\
    # TODO: convert to same unit
    # metadata["AgingTime"] = remove_units(metadata["AgingTime"])

    # format aging temp
    metadata["AgingTemp"] = remove_units(metadata["AgingTemp"])

    # format AgingHumidity, AgingOxygen, Impurity, ImpurityConcentration
    # TODO

    # format detector, done in tif metadata section

    # format coating
    # TODO

    # format particle
    particle_str = metadata["Particle"].lower()
    particle_str = particle_str.replace("particle", "").replace("part", "")

    if particle_str.isdigit() and (int(particle_str) <= 26):
        particle_str = chr(int(particle_str) + 96)

    metadata["Particle"] = "part" + particle_str

    # format replicate
    rep_str = metadata["Replicate"].lower()
    rep_str = rep_str.replace("replicate", "rep")
    if rep_str.find("rep") == -1:
        rep_str = "rep" + rep_str

    metadata["Replicate"] = rep_str

    # format image
    # format date
    # TODO: on some images, these fields are flipped. Get image capture date from metadata

    # read tif metadata
    metadata["DetectorMode"] = "NA"  # place this if file read breaks
    try:
        with tifffile.TiffFile(full_filename) as tif:
            detector_name = tif.fei_metadata["Detectors"]["Name"]
            detector_mode = tif.fei_metadata[detector_name]["Signal"]
            try:
                system_name = tif.fei_metadata["System"]["SystemType"]
            except KeyError:
                system_name = "unknown"

            # format detector mode
            if detector_mode in ["BSE", "SE"]:
                metadata["DetectorMode"] = detector_mode
            # if detector col hits the right value:
            elif metadata["Detector"] in ["BSE", "SE"]:
                metadata["DetectorMode"] = metadata["Detector"]
            # Schwerdt '18 images are all SE, but some are labeled "CN"
            elif detector_mode == "CN":
                metadata["DetectorMode"] = "SE"
            # if the detector is Teneo we can search T1/T2
            elif (system_name.find("Teneo") != -1) or (metadata["Detector"].find("Teneo") != -1):
                if detector_name == "T1" or (metadata["Detector"].find("T1") != -1):
                    metadata["DetectorMode"] = "BSE"
                elif detector_name == "T2" or (metadata["Detector"].find("T2") != -1):
                    metadata["DetectorMode"] = "SE"
            # if we can't sus it out
            else:
                # warn("unable to find SEM mode")
                metadata["DetectorMode"] = "NA"

            # format detector
            if (system_name.find("Teneo") != -1) or (metadata["Detector"].find("Teneo") != -1):
                metadata["Detector"] = "Teneo"
            # elif (metadata["Detector"] == "TLD") and detector_mode in ["CN", "SE"]:
            #     metadata["Detector"] = "Nova"
            elif (system_name.find("Helios") != -1) or (metadata["Detector"].find("Helios") != -1):
                metadata["Detector"] = "Helios"
            elif (system_name.find("Quattro") != -1) or (metadata["Detector"].find("Quattro") != -1):
                metadata["Detector"] = "Quattro"
            elif (system_name.find("Quanta") != -1) or (metadata["Detector"].find("Quanta") != -1):
                metadata["Detector"] = "Quanta"
            elif (
                (system_name.find("Nova") != -1)
                or (metadata["Detector"].find("Nova") != -1)
                or (tif.fei_metadata["System"]["Dnumber"] == "D8217")
            ):
                metadata["Detector"] = "Nova"
            else:
                # warn("unable to find SEM detector")
                metadata["Detector"] = "NA"

            # format HFW
            metadata["HFW"] = round(tif.fei_metadata["EBeam"]["HFW"] * 1e6, 2)
    except tifffile.TiffFileError:
        # warn(f"{full_filename} could not be read")
        pass
    except KeyError:
        # warn(f"{full_filename} has a metadata key error")
        pass
    except TypeError:
        # warn(f"{full_filename} has no fei metadata")
        pass

    # if hfw not updated, remove units
    if type(metadata["HFW"]) == str:
        metadata["HFW"] = float(remove_units(metadata["HFW"]))

    # copy filename
    metadata["FileName"] = full_filename

    # generate file hash
    metadata["Hash"] = get_hash(full_filename)

    # cut off infobar
    match metadata["Detector"]:
        case "Helios":
            dbh = 79
        case "Teneo":
            dbh = 46
        case "Nova":
            dbh = 60
        case "Quanta":
            dbh = 59
        case "Quattro":
            dbh = 0
        case _:  # catch mistakes or errors
            dbh = 0
    metadata["Databar_Height"] = dbh

    return metadata


def filter_dataframe(dataframe: pd.DataFrame, filters) -> pd.DataFrame:
    """
    Filter dataframe by given filters
    """
    for attribute in filters:
        values = filters[attribute]

        if isinstance(values, tuple):
            # filter by range
            assert values[0] < values[1]

            dataframe = dataframe[(dataframe[attribute] >= values[0]) & (dataframe[attribute] <= values[1])]

        elif isinstance(values, list):
            # filter by list
            dataframe = dataframe[dataframe[attribute].isin(values)]
        else:
            error("Invalid filter type")

    return dataframe


def make_lmdb(root: str, name: str, dataframe: pd.DataFrame, configs: dict, type: str = "trainval"):
    # make dirpath
    dirpath = os.path.join(root, name)
    os.makedirs(dirpath, exist_ok=True)

    with px.Writer(dirpath=dirpath, map_size_limit=64000) as db:
        for _, sample in tqdm(dataframe.iterrows(), dynamic_ncols=True, total=len(dataframe)):
            if type == "trainval":
                label = np.array([int(sample["Label"])])
            else:  # if ood
                label = np.array([-1])

            image = np.array(Image.open(sample["FileName"]))

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
            if sample["Detector"] == "Helios":
                databar_height = 79
            elif sample["Detector"] == "Teneo":
                databar_height = 46
            elif sample["Detector"] == "Nova":
                databar_height = 60
            image = image[:, :-databar_height, :]

            N = configs["image_size"]
            images = []

            match configs["crop_type"]:
                case "random":
                    # randomly get NxN patches
                    for _ in range(10):
                        images.append(random_crop(image, (N, N)))

                case "grid":
                    # cut into NxN patches
                    for i in range(
                        (image.shape[1] % N) // 2,
                        image.shape[1] - ((image.shape[1] % N) // 2) - 1,
                        N,
                    ):
                        for j in range(0, image.shape[2], N):
                            images.append(image[:, i : i + N, j : j + N])

                case "whole":
                    images.append(image)

                case _:
                    error("Invalid crop type")

            if np.max(image) < 100:
                warn(f"{sample['FileName']} has max value of {np.max(image)}")

            db.put_samples(
                "label",
                np.repeat(label, len(images), axis=0),
                "image",
                np.array(images),
            )


def make_imagefolder(root: str, name: str, dataframe: pd.DataFrame, configs: dict, type: str = "trainval"):
    # make dirpath
    dirpath = os.path.join(root, name)
    os.makedirs(dirpath, exist_ok=True)

    # cols should be "filename", "label", "labelstr", "original_filename"
    image_metadata = []

    iterrows = tqdm(dataframe.iterrows(), dynamic_ncols=True, total=len(dataframe))
    for _, sample in iterrows:
        if type == "trainval":
            label = int(sample["Label"])
            labelstr = sample["LabelStr"]
        else:  # if ood
            label = -1
            labelstr = "ood"

        image = np.array(Image.open(sample["FileName"]))

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
        if sample["Detector"] == "Helios":
            databar_height = 79
        elif sample["Detector"] == "Teneo":
            databar_height = 46
        elif sample["Detector"] == "Nova":
            databar_height = 60
        image = image[:, :-databar_height, :]

        N = configs["image_size"]
        images = []

        match configs["crop_type"]:
            case "random":
                # randomly get NxN patches
                for _ in range(10):
                    images.append(random_crop(image, (N, N)))

            case "grid":
                # cut into NxN patches
                for i in range(
                    (image.shape[1] % N) // 2,
                    image.shape[1] - ((image.shape[1] % N) // 2) - 1,
                    N,
                ):
                    for j in range(0, image.shape[2], N):
                        images.append(image[:, i : i + N, j : j + N])

            case "whole":
                images.append(image)

            case _:
                error("Invalid crop type")

        for i, img in enumerate(images):
            new_filename = f"{sample['Hash'][:16]}-{labelstr}-{i}.png"
            Image.fromarray(img.transpose((1, 2, 0))).save(os.path.join(dirpath, new_filename))
            image_metadata.append([new_filename, label, labelstr, sample["FileName"]])

    df = pd.DataFrame(image_metadata, columns=["filename", "label", "labelstr", "original_filename"])
    df.to_csv(os.path.join(dirpath, "metadata.csv"), index=False)


def format_material(value):
    """uniform-ize the material name"""

    # TODO: deal with UO2-i/d and AUC-i/d stuff

    if value in [
        "UO2",
        "UO2-Indirect",
        "UO2-indirect",
        "UO2-Direct",
        "UO2-direct",
        "UO2-Reduction",
        "UO2-steam",
    ]:
        return "UO2"
    # TODO: check if a is alpha or amorphous
    if value in [
        "alphaUO3",
        "Alpha-UO3",
        "AlphaUO3",
        "a-UO3",
        "A-UO3",
        "aUO3",
        "Contaminant-A-UO3",
        "UO3",
    ]:
        return "alphaUO3"
    if value in ["AmUO3", "amUO3", "Am-UO3", "AUO3"]:
        return "amorphousUO3"
    if value in ["UO4-2H2O", "UO4"]:
        return "UO4"
    if value in ["U3O8", "238U3O8", "Alpha-U3O8", "233U3O8"]:
        return "U3O8"
    if value in ["Beta-UO3"]:
        return "betaUO3"
    if value in ["Na6U7O24"]:
        return "SDU"

    return value


def format_starting_material(value):
    """uniform-ize the starting material name"""

    if value in ["SDU", "dSDU", "eSDU", "nSDU"]:
        return "SDU"
    if value in ["dMDU", "eMDU", "nMDU"]:
        return "MDU"
    if value in [
        "UO4-2H2O",
        "UO4",
    ]:
        return "UO4"
    if value in [
        "UO4-washed",
        "washedUO4",
        "WashedUO4",
    ]:
        return "washedUO4"
    if value in [
        "unwashedUO42H2O",
        "UO4-unwashed",
        "UnwashedUO4-2H2O",
        "UnwashedUO4-2H2O-direct",
    ]:
        return "unwashedUO4"
    # if value in ["AUCd"]:
    #     return "AUCd"
    # if value in ["AUCi"]:
    #     return "AUCi"
    # if value in ["MDU"]:
    #     return "MDU"
    # if value in ["Umetal"]:
    #     return "Umetal"
    # TODO: what to do about mixtures?

    return value


def remove_units(value):
    """remove units from numeric values"""
    if value == "NA":
        return value

    unitless = re.findall(r"([0-9]+\.?[0-9]*|\.[0-9]+)", value)
    return ".".join(unitless)


def get_hash(filename):
    """get sha256 hash for given file"""
    sha256 = hashlib.sha256()

    with open(filename, "rb") as file:
        while True:
            data = file.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)

    return sha256.hexdigest()


def print_green(string: str):
    """print in green"""
    print(f"\033[92m{string}\033[00m")


def warn(string: str):
    """print warn in yellow"""
    print(f"\033[93mWARN:\033[00m {string}")


def error(string: str):
    """print error in red"""
    print(f"\033[91mERROR:\033[00m {string}")
