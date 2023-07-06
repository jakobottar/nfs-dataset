"""
read image files from folder and parse their metadata stored in filename

Jakob Johnson - 02-28-2023
"""
import multiprocessing
import os

import pandas as pd
import tifffile

from utils import format_material, format_starting_material, get_hash, remove_units

ROOT = "/scratch_nvme/jakobj/all-morpho-images"
# ROOT = "/scratch_nvme/jakobj/multimag/U3O8ADU/10000x"


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

    ### Reformatting block:

    # handle formatting
    metadata["Material"] = format_material(metadata["Material"])
    metadata["StartingMaterial"] = format_starting_material(
        metadata["StartingMaterial"]
    )
    metadata["Magnification"] = remove_units(metadata["Magnification"])
    metadata["CalcinationTemp"] = remove_units(metadata["CalcinationTemp"])
    # metadata["CalcinationTime"] = remove_units(metadata["CalcinationTime"])
    # metadata["AgingTime"] = remove_units(metadata["AgingTime"])
    metadata["AgingTemp"] = remove_units(metadata["AgingTemp"])

    # standardize particle formatting
    particle_str = metadata["Particle"].lower()
    particle_str = particle_str.replace("particle", "").replace("part", "")

    if particle_str.isdigit() and (int(particle_str) <= 26):
        particle_str = chr(int(particle_str) + 96)

    metadata["Particle"] = "part" + particle_str

    # standardize replicate formatting
    rep_str = metadata["Replicate"].lower()
    rep_str = rep_str.replace("replicate", "rep")
    if rep_str.find("rep") == -1:
        rep_str = "rep" + rep_str

    metadata["Replicate"] = rep_str

    try:
        with tifffile.TiffFile(os.path.join(ROOT, filename)) as tif:
            detector_name = tif.fei_metadata["Detectors"]["Name"]
            detector_mode = tif.fei_metadata[detector_name]["Signal"]
            try:
                system_name = tif.fei_metadata["System"]["SystemType"]
            except KeyError:
                system_name = "unknown"

            # if detector mode hits the right value:
            if detector_mode in ["BSE", "SE"]:
                metadata["DetectorMode"] = detector_mode
            # if detector col hits the right value:
            elif metadata["Detector"] in ["BSE", "SE"]:
                metadata["DetectorMode"] = metadata["Detector"]
            # Schwerdt '18 images are all SE, but some are labeled "CN"
            elif detector_mode == "CN":
                metadata["DetectorMode"] = "SE"
            # if the detector is Teneo we can search T1/T2
            elif (system_name.find("Teneo") != -1) or (
                metadata["Detector"].find("Teneo") != -1
            ):
                if detector_name == "T1" or (metadata["Detector"].find("T1") != -1):
                    metadata["DetectorMode"] = "BSE"
                elif detector_name == "T2" or (metadata["Detector"].find("T2") != -1):
                    metadata["DetectorMode"] = "SE"
            # if we can't sus it out
            else:
                print("bad mode")
                metadata["DetectorMode"] = "NA"

            # get detector system
            if (system_name.find("Teneo") != -1) or (
                metadata["Detector"].find("Teneo") != -1
            ):
                metadata["Detector"] = "Teneo"
            # elif (metadata["Detector"] == "TLD") and detector_mode in ["CN", "SE"]:
            #     metadata["Detector"] = "Nova"
            elif (system_name.find("Helios") != -1) or (
                metadata["Detector"].find("Helios") != -1
            ):
                metadata["Detector"] = "Helios"
            elif (system_name.find("Quattro") != -1) or (
                metadata["Detector"].find("Quattro") != -1
            ):
                metadata["Detector"] = "Quattro"
            elif (system_name.find("Quanta") != -1) or (
                metadata["Detector"].find("Quanta") != -1
            ):
                metadata["Detector"] = "Quanta"
            elif (
                (system_name.find("Nova") != -1)
                or (metadata["Detector"].find("Nova") != -1)
                or (tif.fei_metadata["System"]["Dnumber"] == "D8217")
            ):
                metadata["Detector"] = "Nova"
            else:
                print("bad detector")
                metadata["Detector"] = "NA"

            metadata["HFW"] = round(tif.fei_metadata["EBeam"]["HFW"] * 1e6, 2)
    except tifffile.TiffFileError:
        print(f"{filename} is not a readable tiff file")
    except:
        print(f"{filename} has some other error")

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
    # dupes = df[df.duplicated(subset=["Hash"], keep=False)]
    # dupes.to_csv("./dupes.csv", index=False)

    pre = len(df)
    deduped = df.drop_duplicates(subset="Hash", keep="first")
    print(f"I found {pre - len(deduped)} duplicate images")
    deduped.to_csv("./deduped.csv", index=False)

    # for col in [
    #     "Material",
    #     # "Magnification",
    #     # "Resolution",
    #     # "HFW",
    #     "StartingMaterial",
    #     # "CalcinationTemp",
    #     # "CalcinationTime",
    #     # "AgingTime",
    #     # "AgingTemp",
    #     # "AgingHumidity",
    #     # "AgingOxygen",
    #     # "Impurity",
    #     # "ImpurityConcentration",
    #     # "Detector",
    #     # "Coating",
    #     # TODO: fix these
    #     # "Replicate",
    #     # "Particle",
    #     # "Image",
    #     # "Date",
    # ]:
    #     print(col)
    #     print(deduped[col].unique())
