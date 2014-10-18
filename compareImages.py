from os import listdir
from os.path import isfile, join
from PIL import Image, ImageChops

def equal(im1, im2):
    try:
        return ImageChops.difference(im1, im2).getbbox() is None
    except ValueError:
        print("Images do not match.")

dupDir = "/home/airalcorn2/art_duplicates/"
dups = [f for f in listdir(dupDir) if isfile(join(dupDir, f))]

for i in range(0, len(dups) - 1):
    print(i)
    for j in range(i + 1, len(dups)):
        if equal(Image.open(dupDir + dups[i]), Image.open(dupDir + dups[j])):
            print(dups[i] + " " + dups[j])