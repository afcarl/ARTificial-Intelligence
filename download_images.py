#!/usr/bin/env python3

import csv
import os
import urllib3

from lxml import html

ART_DIR = "art_images/"
f = open("ART_DIR", "w")
f.write(ART_DIR)
f.close()

distinct_image_names = set()


def download_image(url):
    global distinct_image_names
    
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

    (base_image_name, ext) = image_url.split("/")[-1].split(".")
    image_name_count = 0
    image_name = "{0}_{1}.{2}".format(base_image_name, image_name_count, ext)
    # Need to check for duplicate image names.
    while image_name in distinct_image_names:
        image_name_count += 1
        image_name = base_image_name + str(image_name_count)

    timeout = True
    while timeout is True:
        try:
            http = urllib3.PoolManager()
            image_src = http.urlopen("GET", "http://www.wga.hu" + image_url, timeout = 5)
            output = open(ART_DIR + image_name, "wb")
            output.write(image_src.data)
            output.close()
            timeout = False
            distinct_image_names.add(image_name)
            return image_name
        except urllib3.exceptions.ConnectTimeoutError:
            print("timeout")


def main():
    os.makedirs(ART_DIR, exist_ok = True)
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
        title = row["TITLE"]
        row["IMAGE"] = download_image(url)
        writer.writerow(row)


if __name__ == "__main__":
    main()