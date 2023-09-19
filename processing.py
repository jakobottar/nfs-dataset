import multiprocessing
import os

import pandas as pd
import pyxis as px
import tifffile

import utils


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
    metadata["Material"] = utils.format_material(metadata["Material"])

    # format magnification
    metadata["Magnification"] = utils.remove_units(metadata["Magnification"])

    # format resolution
    # TODO

    # format HFW, done in tif metadata section

    # format starting material
    metadata["StartingMaterial"] = utils.format_starting_material(metadata["StartingMaterial"])

    # format calcination temp
    metadata["CalcinationTemp"] = utils.remove_units(metadata["CalcinationTemp"])

    # format calcination time
    # TODO: convert to same unit
    # metadata["CalcinationTime"] = remove_units(metadata["CalcinationTime"])

    # format aging time\
    # TODO: convert to same unit
    # metadata["AgingTime"] = remove_units(metadata["AgingTime"])

    # format aging temp
    metadata["AgingTemp"] = utils.remove_units(metadata["AgingTemp"])

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
                # utils.warn("unable to find SEM mode")
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
                # utils.warn("unable to find SEM detector")
                metadata["Detector"] = "NA"

            # format HFW
            metadata["HFW"] = round(tif.fei_metadata["EBeam"]["HFW"] * 1e6, 2)
    except tifffile.TiffFileError:
        # utils.warn(f"{full_filename} could not be read")
        pass
    except KeyError:
        # utils.warn(f"{full_filename} has a metadata key error")
        pass
    except TypeError:
        # utils.warn(f"{full_filename} has no fei metadata")
        pass

    # if hfw not updated, remove units
    if type(metadata["HFW"]) == str:
        metadata["HFW"] = float(utils.remove_units(metadata["HFW"]))

    # copy filename
    metadata["FileName"] = full_filename

    # generate file hash
    metadata["Hash"] = utils.get_hash(full_filename)

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
            utils.error("Invalid filter type")

    return dataframe
