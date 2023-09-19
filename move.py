import os

import pandas as pd
import skimage

ORIGIN = "/scratch_nvme/jakobj/all-morpho-images"
DESTINATION = "/scratch_nvme/jakobj/multimag"

# df = pd.read_csv("./mount.csv")
# df = pd.read_csv("./teneo.csv")
df = pd.read_csv("./impurities.csv")
print(df)

for i, row in df.iterrows():
    print(row["FileName"])
    newdir = os.path.join(
        DESTINATION,
        "ood_impurities"
        # f"{row['Material']}{row['StartingMaterial']}",
        # row["Detector"],
        # row["DetectorMode"],
        # str(row["Magnification"]) + "x",
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
    databar_height = 0
    if row["Detector"] == "Helios":
        databar_height = 79
    elif row["Detector"] == "Teneo":
        databar_height = 46
    elif row["Detector"] == "Nova":
        databar_height = 60
    image = image[:-databar_height, :]

    # cut into fours
    width, height = image.shape[0], image.shape[1]
    image_0 = image[: width // 2, : height // 2]
    image_1 = image[width // 2 :, : height // 2]
    image_2 = image[: width // 2, height // 2 :]
    image_3 = image[width // 2 :, height // 2 :]

    # save to new location
    newfilename = os.path.join(newdir, f"image_{i:03d}")
    skimage.io.imsave(f"{newfilename}_0.tif", image_0)
    skimage.io.imsave(f"{newfilename}_1.tif", image_1)
    skimage.io.imsave(f"{newfilename}_2.tif", image_2)
    skimage.io.imsave(f"{newfilename}_3.tif", image_3)
