#!/usr/bin/env python3

import csv
import pickle


def levenshtein(s1, s2):
    """
    From http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python.
    :param s1:
    :param s2:
    :return:
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for (i, c1) in enumerate(s1):
        current_row = [i + 1]
        for (j, c2) in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_painting_artists():
    reader = csv.DictReader(open("catalog_final.csv"), delimiter = ";")

    artists = set()

    for row in reader:
        artist = row["AUTHOR"]
        artists.add(artist)

    artists = list(artists)
    artists.sort()

    artists_lower_case = []
    for artist in artists:
        if "," in artist:
            artist_parts = artist.split(", ")
            new_artist = artist_parts[1] + " " + artist_parts[0]
            artists_lower_case.append(new_artist.lower())
        else:
            artists_lower_case.append(artist.lower())

    artists_lower_case.sort()
    return artists_lower_case


def main():
    artist_to_movement = pickle.load(open("artist_to_movement.pydict"))
    movement_artists = artist_to_movement.keys()
    painting_artists = get_painting_artists()

    potential_matches = {movement_artist.lower() for movement_artist in movement_artists}
    artist_match = {} # pickle.load(open("artist_match.pydict"))

    for (i, artist) in enumerate(painting_artists):
        # print(artist)
        closest_matches = []
        min_edit_dist = 0

        if artist not in potential_matches:
            min_edit_dist = float("inf")
            for potential_match in potential_matches:
                dist = levenshtein(artist, potential_match)
                if dist == min_edit_dist:
                    closest_matches.append(potential_match)
                elif dist < min_edit_dist:
                    min_edit_dist = dist
                    closest_matches = [potential_match]
        else:
            closest_matches = [artist]

        print("Number: {0}".format(i))
        print("{0} left.\n".format(len(painting_artists) - 1))
        print("{0} {1}".format(artist, min_edit_dist))
        closest_matches.sort()
        print(closest_matches)

        try_again = True
        while try_again:
            try:
                selection = 0
                if artist not in potential_matches:
                    selection = int(input("Pick one or none: "))
                if 0 <= selection < len(closest_matches):
                    artist_match[artist] = closest_matches[selection]
                    potential_matches.remove(closest_matches[selection])
                    pickle.dump(artist_match, open("artist_match.pydict", "w"))
                    try_again = False
                else:
                    raise ValueError
            except SyntaxError:
                try_again = False
            except ValueError:
                print("Invalid selection.")
        print("\n")


if __name__ == "__main__":
    main()