import cPickle
import csv

artistCounts = {}

reader = csv.DictReader(open("artistsData.csv"), delimiter = ";")
artists_raw_data = []

for row in reader:
    artist = row["ARTIST"]
    if artist in artistCounts:
        artistCounts[artist] += 1
    else:
        artistCounts[artist] = 1

artistCounts = artistCounts.items()
artistCounts.sort(key = lambda artist: artist[1], reverse = True)

topArtists = {}
for artist in artistCounts[:100]:
    topArtists[artist[0]] = True

cPickle.dump(topArtists, open("topArtists.pydict", "w"))