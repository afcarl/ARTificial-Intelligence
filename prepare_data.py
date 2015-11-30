#!/usr/bin/env python3

import csv
import numpy as np
import pickle
import random

from PIL import Image


ART_DIR = open("ART_DIR").read().strip()
TOP_ARTISTS = 100
TOP_MOVEMENTS = 14


def dump_counts(column, n = TOP_ARTISTS):
    if column != "artist":
        n = TOP_MOVEMENTS

    capitalized_column = column.upper()
    counts = {}
    reader = csv.DictReader(open("{0}s_data.csv".format(column)), delimiter = ";")
    for row in reader:
        value = row[capitalized_column]
        counts[value] = counts.get(value, 0) + 1

    counts = list(counts.items())
    counts.sort(key = lambda item_count: item_count[1], reverse = True)

    top_counts = {item[0] for item in counts[:n]}
    return top_counts


def scale_down_by_width(base_width, base_height, img):
    
    (width, height) = img.size
    wpercent = (base_width / float(width))
    hsize = int((float(height) * float(wpercent)))
    img = img.resize((base_width, hsize), Image.ANTIALIAS)
    
    (new_width, new_height) = img.size
    shave = (new_height - base_height) / 2
    if (new_height - base_height) % 2 == 1:
        img = img.crop((0, shave, new_width, new_height - (shave + 1)))
    else:
        img = img.crop((0, shave, new_width, new_height - shave))
    return img


def scale_down_by_height(base_width, base_height, img):
    
    (width, height) = img.size
    hpercent = (base_height / float(height))
    wsize = int((float(width) * float(hpercent)))
    img = img.resize((wsize, base_height), Image.ANTIALIAS)
    
    (new_width, new_height) = img.size
    shave = (new_width - base_width) / 2
    if (new_width - base_width) % 2 == 1:
        img = img.crop((shave, 0, new_width - (shave + 1), new_height))
    else:
        img = img.crop((shave, 0, new_width - shave, new_height))
    return img


def get_raw_data(column):
    
    min_width = float("inf")
    min_height = float("inf")
    
    widths = []
    heights = []
    
    capitalized_column = column.upper()
    reader = csv.DictReader(open("{0}s_data.csv".format(column)), delimiter = ";")
    raw_data = []
    top_counts = dump_counts(column)
    
    for row in reader:
        value = row[capitalized_column]
        image = row["IMAGE"]
        if value in top_counts:
            raw_data.append((value, image))
            try:
                im = Image.open(ART_DIR + image)
                (width, height) = im.size
                widths.append(width)
                heights.append(height)
                if width < min_width:
                    min_width = width
                if height < min_height:
                    min_height = height
            except IOError:
                print(image)
    
    random.shuffle(raw_data)
    return raw_data


def get_data_and_labels(raw_data, min_width, min_height, value_to_int = {},
                        value_int = 0, training = False):

    data = []
    labels = []

    for (i, sample) in enumerate(raw_data):

        if i % 1000 == 0:
            print(i)

        value = sample[0]
        img = sample[1]

        if training and value not in value_to_int:
            value_to_int[value] = value_int
            value_int += 1

        if not training and value not in value_to_int:
            raise ValueError("Unseen value: {0}".format(value))

        value_int = value_to_int[value]

        im = Image.open(ART_DIR + img)
        (width, height) = im.size
        if height > width:
            im = scale_down_by_width(min_width, min_height, im)
        else:
            im = scale_down_by_height(min_width, min_height, im)

        (width, height) = im.size

        img = np.asarray(im, dtype = "float64") / 256.
        # Put image in 4D tensor of shape (1, 3, height, width).
        try:
            img_ = img.swapaxes(0, 2).swapaxes(1, 2)
            data.append(img_)
            labels.append(value_int)
        except:
            # Probably grayscale.
            print(sample[1])

    data = np.array(data)
    labels = np.array(labels)
    data_and_labels = (data, labels)
    if training:
        return (data_and_labels, value_to_int)
    else:
        return data_and_labels


def prepare_data_sets(raw_data, column):
    cutoff = 0.8
    
    train_size = int(cutoff * len(raw_data))
    train_data = raw_data[:train_size]
    test_data = raw_data[train_size:]
    
    new_train_size = int(cutoff * len(train_data))
    valid_data = train_data[new_train_size:]
    train_data = train_data[:new_train_size]
    
    """
    batch_size = len(train_data) / 3
    
    (batch_1, value_to_int) = get_data_and_labels(train_data[:batch_size], training = True)
    pickle.dump(batch_1, open("batch_1.numpy", "w"))
    """
    
    min_width = 100
    min_height = 100
    
    (train_set, value_to_int) = get_data_and_labels(train_data, min_width,
                                                    min_height, training = True)
    pickle.dump(value_to_int, open("{0}_to_int.pydict".format(column), "w"))
    pickle.dump(train_set, open("train_set_{0}.numpy".format(column), "w"))
    
    valid_set = get_data_and_labels(valid_data, min_width, min_height,
                                    value_to_int)
    pickle.dump(valid_set, open("valid_set_{0}.numpy".format(column), "w"))
    
    test_set = get_data_and_labels(test_data, min_width, min_height,
                                   value_to_int)
    pickle.dump(test_set, open("test_set_{0}.numpy".format(column), "w"))
    
    all_labels = train_set[1].tolist() + valid_set[1].tolist() + test_set[1].tolist()
    counts = {}
    for label in all_labels:
        counts[label] = counts.get(label, 0) + 1
    
    counts = list(counts.items())
    counts.sort(key = lambda artist: artist[1], reverse = True)
    
    baseline = float(counts[0][1]) / float(len(all_labels))
    print(baseline)


def main():
    raw_data = get_raw_data("artist")
    # raw_data = raw_data[:100] # For testing.
    prepare_data_sets(raw_data, "artist")
    
    raw_data = get_raw_data("movement")
    prepare_data_sets(raw_data, "movement")


if __name__ == "__main__":
    main()