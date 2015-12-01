#!/usr/bin/env python3

import csv
import os
import urllib3

from lxml import html
from os import listdir
from os.path import isfile, join

ART_DIR = "art_images/"
f = open("ART_DIR.txt", "w")
f.write(ART_DIR)
f.close()

bad_urls = open("bad_urls.txt", "w")


def download_image(url, image_name):
    doc = None
    fail = True
    while fail:
        try:
            doc = html.parse(url).getroot()
            fail = False
        except IOError:
            print("IOError")
    
    image_url = ""
    
    for (elem, attribute, link, pos) in doc.iterlinks():
        if attribute == "href" and elem.tag == "a" and link.endswith(".jpg"):
            image_url = link

    if image_url == "":
        print("Bad URL.")
        bad_urls.write(url + "\n")
        return False

    timeout = True
    while timeout:
        try:
            http = urllib3.PoolManager()
            image_src = http.urlopen("GET", "http://www.wga.hu" + image_url, timeout = 5)
            output = open(ART_DIR + image_name, "wb")
            output.write(image_src.data)
            output.close()
            return True
        except urllib3.exceptions.ConnectTimeoutError:
            print("timeout")


def main():
    os.makedirs(ART_DIR, exist_ok = True)
    art_images = [f for f in listdir(ART_DIR) if isfile(join(ART_DIR, f))]
    already_downloaded = {art_image for art_image in art_images}
    distinct_names = set()

    reader = csv.DictReader(open("catalog.csv", encoding = "iso-8859-1"), delimiter = ";")
    writer_fieldnames = reader.fieldnames + ["IMAGE"]
    writer = csv.DictWriter(open("catalog_final.csv", "w", encoding = "utf-8"),
                            fieldnames = writer_fieldnames, delimiter = ";")
    writer.writeheader()

    for row in reader:
        if row["FORM"] != "painting":
            continue
        url = row["URL"]
        print(url)

        base_image_name = url.split("/")[-1].replace(".html", "")
        image_name_count = 0
        image_name = "{0}_{1}.jpg".format(base_image_name, image_name_count)
        # Need to check for duplicate image names.
        while image_name in distinct_names:
            image_name_count += 1
            image_name = "{0}_{1}.jpg".format(base_image_name, image_name_count)

        # Also need to check for images that have already been downloaded.
        if image_name in already_downloaded or download_image(url, image_name):
            row["IMAGE"] = image_name
            writer.writerow(row)
            already_downloaded.add(image_name)
            distinct_names.add(image_name)

    bad_urls.close()


if __name__ == "__main__":
    main()