import cPickle
import csv

movementCounts = {}

reader = csv.DictReader(open("movementsData.csv"), delimiter = ";")
movements_raw_data = []

for row in reader:
    movement = row["MOVEMENT"]
    if movement in movementCounts:
        movementCounts[movement] += 1
    else:
        movementCounts[movement] = 1

movementCounts = movementCounts.items()
movementCounts.sort(key = lambda movement: movement[1], reverse = True)

topMovements = {}
for movement in movementCounts[:14]:
    topMovements[movement[0]] = True

cPickle.dump(topMovements, open("topMovements.pydict", "w"))