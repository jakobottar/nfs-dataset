import os
import shutil

import numpy as np
import pandas as pd
import skimage
import tifffile

ORIGIN = "/scratch_nvme/jakobj/all-morpho-images"
DESTINATION = "/scratch_nvme/jakobj/multimag"

df = pd.read_csv("./filtered.csv")
print(df)

for _, row in df.iterrows():
    print(row["FileName"])
    newdir = os.path.join(
        DESTINATION,
        row["Detector"],
        row["DetectorMode"],
        f"{row['Material']}{row['StartingMaterial']}",
        str(row["Magnification"]) + "x",
    )

    if not os.path.exists(newdir):
        os.makedirs(newdir)

    oldfilename = os.path.join(ORIGIN, row["FileName"])

    # open image
    image = skimage.io.imread(oldfilename, as_gray=False)

    # convert to RGB if needed
    if image.shape[-1] != 3:
        image = skimage.color.gray2rgb(image)

    # cut off infobar
    if row["Detector"] == "Helios":
        databar_height = 79
    elif row["Detector"] == "Teneo":
        databar_height = 46
    image = image[:-databar_height, :]

    # cut into fours
    width, height = image.shape[0], image.shape[1]
    image_1 = image[: width // 2, : height // 2]
    image_2 = image[width // 2 :, : height // 2]
    image_3 = image[: width // 2, height // 2 :]
    image_4 = image[width // 2 :, height // 2 :]

    # save to new location
    newfilename = os.path.join(newdir, row["FileName"])
    skimage.io.imsave(f"{newfilename[:-4]}_1.tif", image_1)
    skimage.io.imsave(f"{newfilename[:-4]}_2.tif", image_2)
    skimage.io.imsave(f"{newfilename[:-4]}_3.tif", image_3)
    skimage.io.imsave(f"{newfilename[:-4]}_4.tif", image_4)
