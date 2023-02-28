"""
read image files from folder and parse their metadata stored in filename

Jakob Johnson - 02-28-2023
"""
import multiprocessing
import os

import pandas as pd

from utils import format_material, format_starting_material, get_hash, remove_units

ROOT = "/nvmescratch/jakobj/all-morpho-images"


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

    # handle formatting
    metadata["Material"] = format_material(metadata["Material"])
    metadata["StartingMaterial"] = format_starting_material(
        metadata["StartingMaterial"]
    )
    metadata["Magnification"] = remove_units(metadata["Magnification"])
    # metadata["HFW"] = remove_units(metadata["HFW"])
    metadata["CalcinationTemp"] = remove_units(metadata["CalcinationTemp"])
    # metadata["CalcinationTime"] = remove_units(metadata["CalcinationTime"])
    # metadata["AgingTime"] = remove_units(metadata["AgingTime"])
    metadata["AgingTemp"] = remove_units(metadata["AgingTemp"])

    # TODO: fix HFW info
    # TODO: get all TIF metadata

    metadata["FileName"] = filename
    metadata["Hash"] = get_hash(os.path.join(ROOT, filename))

    return metadata


if __name__ == "__main__":

    filenames = os.listdir(ROOT)

    # get image metadata and hashes
    with multiprocessing.Pool(16) as pool:
        image_metadata = pool.map(get_metadata, filenames)

    df = pd.DataFrame().from_dict(image_metadata)
    df.to_csv("./full.csv", index=False)

    # dedupe based on hash
    dupes = df[df.duplicated(subset=["Hash"], keep=False)]
    dupes.to_csv("./dupes.csv", index=False)

    pre = len(df)
    deduped = df.drop_duplicates(subset="Hash", keep="first")
    print(f"I found {pre - len(deduped)} duplicate images")
    deduped.to_csv("./deduped.csv", index=False)

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
        # "Detector",
        # "Coating",
        # TODO: fix these
        # "Replicate",
        # "Particle",
        # "Image",
        # "Date",
    ]:
        print(col)
        print(deduped[col].unique())
