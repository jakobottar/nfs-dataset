"""
check files in directory to see if they match the standard pattern
args:
    --directory or -d: str, directory to search
    --update or -u: flag, update filenames to match scheme
"""
import argparse
import os

from utils import format_material, remove_units

ACCEPTABLE_NAMES = {
    "Material": [
        "UO2",
        "UO2-indirect",
        "UO2-direct",
        "UO3",
        "amUO3",
        "alphaUO3",
        "betaUO3",
        "U3O8",
        "alphaU3O8",
        "UO4",
    ],
    # "Magnification": [],
    # "Resolution": [],
    # "HFW": [],
    # "StartingMaterial": [],
    # "CalcinationTemp": [],
    # "CalcinationTime": [],
    # "AgingTime": [],
    # "AgingTemp": [],
    # "AgingHumidity": [],
    # "AgingOxygen": [],
    # "Impurity": [],
    # "ImpurityConcentration": [],
    # "Detector": [],
    # "Coating": [],
}


def error(string: str):
    """print error in red"""
    print(f"\033[91mERROR:\033[00m {string}")


def warn(string: str):
    """print warn in yellow"""
    print(f"\033[93mWARN:\033[00m {string}")


def get_metadata(filename):
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
    name = filename[:-4]

    name_split = name.split(sep="_")

    # catch Alpha_UO3 and Am_UO3 naming mistakes
    if name_split[0].lower() in ["am", "alpha"]:
        warn("found '_' in material field, attempting to fix...")
        first = name_split[0].lower()
        del name_split[0]
        name_split[0] = first + name_split[0]

    # check number of fields
    if len(name_split) > 19:
        error(f"{filename} has too many fields!")
        return
    if len(name_split) < 19:
        error(f"{filename} has too few fields!")
        return

    # populate metadata struct
    for i, key in enumerate(metadata.keys()):
        metadata[key] = name_split[i]

    # remove units
    metadata["Magnification"] = remove_units(metadata["Magnification"])
    metadata["HFW"] = remove_units(metadata["HFW"])
    metadata["CalcinationTemp"] = remove_units(metadata["CalcinationTemp"])
    metadata["CalcinationTime"] = remove_units(metadata["CalcinationTime"])
    metadata["AgingTime"] = remove_units(metadata["AgingTime"])
    metadata["AgingTemp"] = remove_units(metadata["AgingTemp"])

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory", "-d", type=str, help="directory to search and verify filenames"
    )
    parser.add_argument(
        "--update", "-u", help="update filenames to match scheme", action="store_true"
    )
    FLAGS = parser.parse_args()

    filenames = os.listdir(FLAGS.directory)
    for name in filenames:
        metadata = get_metadata(name)
        if metadata:
            # check if all values are in the standard formats
            for key, valid in ACCEPTABLE_NAMES.items():
                if metadata[key] not in valid:
                    warn(
                        f"found an invalid value '{metadata[key]}' in filename, attempting to fix..."
                    )
                    match key:
                        case "Material":
                            metadata[key] = format_material(metadata[key])
                        case _:
                            error("could not fix!")
                            metadata = None
                            break

                    if metadata[key] is None:
                        error("could not fix!")
                        metadata = None
                        break

        if metadata and FLAGS.update:
            # make new filename from metadata
            new_name = "_".join(metadata.values()) + ".tif"

            # rename file
            old_path = os.path.join(FLAGS.directory, name)
            new_path = os.path.join(FLAGS.directory, new_name)
            # shutil.move(old_path, new_path)
