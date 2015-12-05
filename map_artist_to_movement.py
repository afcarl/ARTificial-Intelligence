#!/usr/bin/env python3

import pickle
import requests

from bs4 import BeautifulSoup

artist_to_movement = {}


def map_artists_to_movement(link, movement):
    global artist_to_movement
    
    url = "http://www.artcyclopedia.com" + link
    r = requests.get(url)
    html = r.text
    soup = BeautifulSoup(html, "lxml")
    
    tables = soup.findChildren("table", cellpadding = 10)
    
    if len(tables) == 0:
        return
    
    artists_table = tables[0]
    rows = artists_table.findChildren("tr")
    
    for row in rows:
        if "artists" not in str(row):
            continue
        cells = row.find_all("td")
        artist = cells[0].find(text = True)
        birth_death = cells[1].find(text = True)
        artist_to_movement[str(artist)] = movement


def main():
    movement_links = open("Links to Movements")
    
    for line in movement_links:
        line = line.strip()
        [link, movement] = line.split("\t")
        map_artists_to_movement(link, movement)
    
    pickle.dump(artist_to_movement, open("artist_to_movement.pydict", "wb"))


if __name__ == "__main__":
    main()