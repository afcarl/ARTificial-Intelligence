#!/usr/bin/env python

import csv
import pickle


def main():
    artist_match = pickle.load(open("artist_match.pydict"))
    artist_to_movement = pickle.load(open("artist_to_movement.pydict"))

    reader = csv.DictReader(open("catalog_final.csv"), delimiter = ";")

    artists_fieldnames = ["ARTIST", "TITLE", "IMAGE"]
    artists_data = csv.DictWriter(open("artists_data.csv", "w"),
                                  artists_fieldnames, delimiter = ";")
    artists_data.writeheader()

    movement_fieldnames = ["ARTIST", "TITLE", "MOVEMENT", "IMAGE"]
    movements_data = csv.DictWriter(open("movements_data.csv", "w"),
                                    movement_fieldnames, delimiter = ";")
    movements_data.writeheader()

    for row in reader:

        artist = row["AUTHOR"]
        title = row["TITLE"]
        image = row["IMAGE"]

        artists_row = {"ARTIST": artist, "TITLE": title, "IMAGE": image}
        artists_data.writerow(artists_row)

        lower_artist = artist.lower()

        if "," in lower_artist:
            artist_parts = lower_artist.split(", ")
            lower_artist = artist_parts[1] + " " + artist_parts[0]

        if lower_artist in artist_match:
            matched_artist = artist_match[lower_artist]
            movement = artist_to_movement[matched_artist]
            movements_row = {"ARTIST": artist, "TITLE": title, "IMAGE": image, "MOVEMENT": movement}
            movements_data.writerow(movements_row)


if __name__ == "__main__":
    main()