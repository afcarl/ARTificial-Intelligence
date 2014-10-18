from bs4 import BeautifulSoup
import cPickle
import requests

artistMovements = {}

def getArtists(link, movement):
    global artistMovements
    
    url = "http://www.artcyclopedia.com" + link
    r = requests.get(url)
    html = r.text
    soup = BeautifulSoup(html)
    
    tables = soup.findChildren("table", cellpadding = 10)
    
    if len(tables) == 0:
        return
    
    artistsTable = tables[0]
    rows = artistsTable.findChildren("tr")
    
    for row in rows:
        if "artists" not in str(row):
            continue
        cells = row.find_all("td")
        artist = cells[0].find(text = True)
        birthDeath = cells[1].find(text = True)
        artistMovements[str(artist)] = movement

movementLinks = open("Links to Movements")

line = movementLinks.readline().strip()

while line != "":
    [link, movement] = line.split("\t")
    getArtists(link, movement)
    line = movementLinks.readline().strip()

cPickle.dump(artistMovements, open("artistMovements.pydict", "w"))
