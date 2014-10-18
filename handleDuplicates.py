import csv
import urllib2

from lxml import html
from os import listdir
from os.path import isfile, join

# Need to check for duplicate image names.

directory = "/home/airalcorn2/art_duplicates/"
dups = [f for f in listdir(directory) if isfile(join(directory, f))]

def downloadImage(url, imageName):
    global directory

    doc = None

    fail = True
    while fail:
        try:
            doc = html.parse(url).getroot()
            fail = False
        except IOError:
            print("IOError")

    imageURL = ""

    for elem, attribute, link, pos in doc.iterlinks():
        if attribute == "href" and elem.tag == "a" and link.endswith(".jpg"):
            imageURL = link

    timeout = True

    while timeout is True:
        try:
            imageSrc = urllib2.urlopen("http://www.wga.hu" + imageURL, timeout = 5)
            imageData = imageSrc.read()
            output = open(directory + imageName, "w")
            output.write(imageData)
            output.close()
            timeout = False
        except:
            print("timeout")

distinctNames = {}
duplicates = {}
reader = csv.DictReader(open("catalog.csv"), delimiter = ";")

for row in reader:
    if row["FORM"] != "painting":
        continue
    url = row["URL"]
    urlParts = url.split("/")
    imageParts = urlParts[-1].split(".")
    imageName = imageParts[0]
    if imageName in distinctNames:
        duplicates[imageName] = True
    else:
        distinctNames[imageName] = True

for imageName in dups:
    [name, ext] = imageName.split(".")
    distinctNames[name] = True

reader = csv.DictReader(open("catalogMod.csv"), delimiter = ";")
fieldNames = reader.fieldnames + ["IMAGE"]
writer = csv.DictWriter(open("catalogFinal.csv", "a"), fieldNames, delimiter = ";")
# writer.writeheader()

for row in reader:
    if row["FORM"] != "painting":
        continue
    newRow = {}
    for key in row.keys():
        newRow[key] = row[key]
    url = row["URL"]
    urlParts = url.split("/")
    imageParts = urlParts[-1].split(".")
    imageName = imageParts[0]
    if imageName not in duplicates:
        newRow["IMAGE"] = imageName + ".jpg"
        writer.writerow(newRow)
    else:
        i = 0
        imageName += str(i)
        while imageName in distinctNames:
            numLen = len(str(i))
            i += 1
            imageName = imageName[:-numLen] + str(i)
        newRow["IMAGE"] = imageName + ".jpg"
        print(url)
        downloadImage(url, imageName + ".jpg")
        writer.writerow(newRow)
        distinctNames[imageName] = True