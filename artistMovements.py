import cPickle
import csv

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

reader = csv.DictReader(open("catalogFinal.csv"), delimiter = ";")

artists = {}

for row in reader:
    artist = row["AUTHOR"]
    if artist not in artists:
        artists[artist] = True

artists = artists.keys()
artists.sort()

artistsLowercase = []
for artist in artists:
    if "," in artist:
        artistParts = artist.split(", ")
        newArtist = artistParts[1] + " " + artistParts[0]
        artistsLowercase.append(newArtist.lower())
    else:
        artistsLowercase.append(artist.lower())

artistsLowercase.sort()

artistMovements = cPickle.load(open("artistMovements.pydict"))

movementArtists = artistMovements.keys()
movementArtists.sort()

matches = {}
nonmatches = {}

possibleMatches = {}

for movementArtist in movementArtists:
    possibleMatches[movementArtist.lower()] = True

for artistLowercase in artistsLowercase:
    if artistLowercase in possibleMatches:
        matches[artistLowercase] = True
    else:
        nonmatches[artistLowercase] = True

# catalogToMovementArtistMap = {}
catalogToMovementArtistMap = cPickle.load(open("catalogToMovementArtistMap.pydict"))
nonmatches = nonmatches.keys()
nonmatches.sort()
foundMovementArtists = {}

i = 0

while i < len(nonmatches):
    # print(artist)
    artist = nonmatches[i]
    i += 1
    closestMatches = []
    minEditDis = float("inf")
    for possibleMatch in possibleMatches.keys():
        if possibleMatch in matches or possibleMatch in foundMovementArtists:
            continue
        dist = levenshtein(artist, possibleMatch)
        if dist == minEditDis:
            closestMatches.append(possibleMatch)
        elif dist < minEditDis:
            minEditDis = dist
            closestMatches = [possibleMatch]
    print("Number: " + str(i))
    print(str(len(nonmatches) - i) + " left.\n")
    print(artist + " " + str(minEditDis))
    closestMatches.sort()
    print(closestMatches)
    tryAgain = True
    while tryAgain:
        try:
            selection = int(input("Pick one or none: "))
            if 0 <= selection < len(closestMatches):
                catalogToMovementArtistMap[artist] = closestMatches[selection]
                foundMovementArtists[closestMatches[selection]] = True
                cPickle.dump(catalogToMovementArtistMap, open("catalogToMovementArtistMap.pydict", "w"))
                tryAgain = False
            else:
                raise ValueError
        except SyntaxError:
            tryAgain = False
        except ValueError:
            print("Invalid selection.")
    print("\n")