#!/usr/bin/env bash

wget http://www.wga.hu/database/download/data_txt.zip
unzip data_txt.zip

./download_images.py
./map_artist_to_movement.py
./match_artist_names.py
./create_data_sets.py
./resize_images.py