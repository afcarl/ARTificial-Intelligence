import csv
import urllib

from lxml import html

directory = "/home/airalcorn2/art/"

def downloadImage(url):
    global directory
    doc = html.parse(url).getroot()
    imageURL = ""
    
    for elem, attribute, link, pos in doc.iterlinks():
        if attribute == "href" and elem.tag == "a" and link.endswith(".jpg"):
            imageURL = link
    
    imageName = imageURL.split("/")[-1]
    urllib.urlretrieve("http://www.wga.hu" + imageURL, directory + imageName)

reader = csv.DictReader(open("catalogMod.csv"), delimiter = ";")

for row in reader:
    if row["FORM"] != "painting":
        continue
    url = row["URL"]
    print(url)
    title = row["TITLE"]
    downloadImage(url)