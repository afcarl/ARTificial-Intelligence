import csv
import urllib2

from lxml import html

# Need to check for duplicate image names.

directory = "/home/airalcorn2/art/"

def downloadImage(url):
    global directory
    doc = html.parse(url).getroot()
    imageURL = ""
    
    for elem, attribute, link, pos in doc.iterlinks():
        if attribute == "href" and elem.tag == "a" and link.endswith(".jpg"):
            imageURL = link

    timeout = True
    
    while timeout is True:
        imageName = imageURL.split("/")[-1]
        try:
            imageSrc = urllib2.urlopen("http://www.wga.hu" + imageURL, timeout = 5)
            imageData = imageSrc.read()
            output = open(directory + imageName, "w")
            output.write(imageData)
            output.close()
            timeout = False
        except:
            print("timeout")

reader = csv.DictReader(open("catalogMod.csv"), delimiter = ";")

for row in reader:
    if row["FORM"] != "painting":
        continue
    url = row["URL"]
    print(url)
    title = row["TITLE"]
    downloadImage(url)