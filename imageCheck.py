from os import listdir
from os.path import isfile, join

import csv

nonDupDir = "/home/airalcorn2/art_finished/"
nonDups = [f for f in listdir(nonDupDir) if isfile(join(nonDupDir, f))]

dupDir = "/home/airalcorn2/art_duplicates/"
dups = [f for f in listdir(dupDir) if isfile(join(dupDir, f))]

allImages = nonDups + dups
images = {}

for image in allImages:
    images[image] = True

duplicates = {}
reader = csv.DictReader(open("catalogFinal.csv"), delimiter = ";")

for row in reader:
    imageName = row["IMAGE"]
    if imageName not in images:
        print("Not found.")
        print(row)
    if imageName in duplicates:
        print("Duplicate.")
        print(row)
    duplicates[imageName] = True