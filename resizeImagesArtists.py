import cPickle
import csv
import numpy as np
import random

from PIL import Image
from os import listdir
from os.path import isfile, join

def scaleDownByWidth(basewidth, baseHeight, img):
    
    (width, height) = img.size
    wpercent = (basewidth / float(width))
    hsize = int((float(height) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    
    newWidth, newHeight = img.size
    shave = (newHeight - baseHeight) / 2
    if (newHeight - baseHeight) % 2 == 1:
        img = img.crop((0, shave, newWidth, newHeight - (shave + 1)))
    else:
        img = img.crop((0, shave, newWidth, newHeight - shave))
    return img

def scaleDownByHeight(basewidth, baseheight, img):
    
    (width, height) = img.size
    hpercent = (baseheight / float(height))
    wsize = int((float(width) * float(hpercent)))
    img = img.resize((wsize, baseheight), Image.ANTIALIAS)
    
    newWidth, newHeight = img.size
    shave = (newWidth - basewidth) / 2
    if (newWidth - basewidth) % 2 == 1:
        img = img.crop((shave, 0, newWidth - (shave + 1), newHeight))
    else:
        img = img.crop((shave, 0, newWidth - shave, newHeight))
    return img

def getDataAndLabels(raw_data, training = False):
    global artistToInt, artistInt, minWidth, minHeight
    
    data = []
    labels = []
    
    i = 0
    
    for sample in raw_data:
        
        if i % 1000 == 0:
            print(i)
        i += 1
        
        artist = sample[0]
        img = sample[1]
        
        if artist not in artistToInt and training:
            artistToInt[artist] = artistInt
            artistInt += 1
        elif artist not in artistToInt and not training:
            print("Unseen artist")
            return
        
        artInt = artistToInt[artist]
        
        im = Image.open("/home/airalcorn2/art_images/" + img)
        (width, height) = im.size
        if height > width:
            im = scaleDownByWidth(minWidth, minHeight, im)
        else:
            im = scaleDownByHeight(minWidth, minHeight, im)
        (width, height) = im.size
        img = np.asarray(im, dtype = 'float64') / 256.
        # put image in 4D tensor of shape (1, 3, height, width)
        try:
            img_ = img.swapaxes(0, 2).swapaxes(1, 2)
            
            data.append(img_)
            labels.append(artInt)
        except:
            # Probably grayscale.
            print(sample[1])
    
    data = np.array(data)
    labels = np.array(labels)
    data_and_labels = (data, labels)
    if training:
        return data_and_labels, artistToInt
    else:
        return data_and_labels

mypath = "/home/airalcorn2/art_images"

minWidth = float("inf")
minHeight = float("inf")

widths = []
heights = []

'''
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for theFile in onlyfiles:
    try:
        im = Image.open(mypath + "/" + theFile)
        (width, height) = im.size
        widths.append(width)
        heights.append(height)
        if width < minWidth:
            minWidth = width
        if height < minHeight:
            minHeight = height
    except IOError:
        print(theFile)
'''

reader = csv.DictReader(open("artistsData.csv"), delimiter = ";")
artists_raw_data = []
topArtists = cPickle.load(open("topArtists.pydict"))

for row in reader:
    artist = row["ARTIST"]
    image = row["IMAGE"]
    if artist in topArtists:
        artists_raw_data.append((artist, image))
        try:
            im = Image.open(mypath + "/" + image)
            (width, height) = im.size
            widths.append(width)
            heights.append(height)
            if width < minWidth:
                minWidth = width
            if height < minHeight:
                minHeight = height
        except IOError:
            print(image)

random.shuffle(artists_raw_data)

# artists_raw_data = artists_raw_data[:100] # For testing.

train_size = int(0.75 * len(artists_raw_data))
train_data = artists_raw_data[:train_size]
new_train_size = int(0.75 * len(train_data))
valid_data = train_data[new_train_size:]
train_data = train_data[:new_train_size]
test_data = artists_raw_data[train_size:]

artistToInt = {}
artistInt = 0

'''
batch_size = len(train_data) / 3

batch_1, artistToInt = getDataAndLabels(train_data[:batch_size], training = True)
cPickle.dump(batch_1, open("batch_1.numpy", "w"))
'''

minWidth = 100
minHeight = 100

train_set, artistToInt = getDataAndLabels(train_data, training = True)
# cPickle.dump(artistToInt, open("artistToInt.pydict", "w"))
# cPickle.dump(train_set, open("train_set.numpy", "w"))

valid_set = getDataAndLabels(valid_data)
# cPickle.dump(valid_set, open("valid_set.numpy", "w"))

test_set = getDataAndLabels(test_data)
# cPickle.dump(test_set, open("test_set.numpy", "w"))

allLabels = train_set[1].tolist() + valid_set[1].tolist() + test_set[1].tolist()
counts = {}
for label in allLabels:
    if label in counts:
        counts[label] += 1
    else:
        counts[label] = 1

counts = counts.items()
counts.sort(key = lambda artist: artist[1], reverse = True)

baseline = float(counts[0][1]) / float(len(allLabels))