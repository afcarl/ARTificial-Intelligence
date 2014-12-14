import cPickle
import csv

catalogToMovementArtistMap = cPickle.load(open("catalogToMovementArtistMap.pydict"))
artistMovements = cPickle.load(open("artistMovements.pydict"))

lowerArtistMovements = {}

for artist in artistMovements.keys():
    lowerArtist = artist.lower()
    lowerArtistMovements[lowerArtist] = artistMovements[artist]

reader = csv.DictReader(open("catalogFinal.csv"), delimiter = ";")

movementsFieldNames = ["ARTIST", "TITLE", "MOVEMENT", "IMAGE"]
movementsData = csv.DictWriter(open("movementsData.csv", "w"), movementsFieldNames, delimiter = ";")
movementsData.writeheader()

artistsFieldNames = ["ARTIST", "TITLE", "IMAGE"]
artistsData = csv.DictWriter(open("artistsData.csv", "w"), artistsFieldNames, delimiter = ";")
artistsData.writeheader()

for row in reader:
    
    artist = row["AUTHOR"]
    title = row["TITLE"]
    image = row["IMAGE"]
    
    artistsRow = {"ARTIST": artist, "TITLE": title, "IMAGE": image}
    
    artistsData.writerow(artistsRow)
    
    lowerArtist = artist.lower()
    
    if "," in lowerArtist:
        artistParts = lowerArtist.split(", ")
        lowerArtist = artistParts[1] + " " + artistParts[0]
    
    if artist in artistMovements or lowerArtist in lowerArtistMovements:
        movement = None
        if artist in artistMovements:
            movement = artistMovements[artist]
        else:
            movement = lowerArtistMovements[lowerArtist]
        movementsRow = {"ARTIST": artist, "TITLE": title, "IMAGE": image, "MOVEMENT": movement}
        movementsData.writerow(movementsRow)