import hashlib
import re

BUF_SIZE = 65536


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

    if value in ["SDU"]:
        return "SDU"
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
