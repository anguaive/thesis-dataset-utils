#!/bin/python3

import argparse
from utils import confirm,  trays_path
from PIL import Image
from pathlib import Path
import sys
import os

parser = argparse.ArgumentParser(description="Crop part images out of tray images and store them in a directory structure.")

parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Perform actions without asking for user confirmation')
args = parser.parse_args()

for tray in trays_path.glob('*'):
    if not (args.force or confirm(f'About to process tray: `{tray.name}`. Continue?')):
        continue

    part_locations = []

    with open(tray / 'tray_descriptor.txt', 'r') as f:
        next(f)
        for line in f:
            part_locations.append(list(map(int, line.split())))

    part_numbers = [row[0] for row in part_locations]
    for part_number in part_numbers:
        try:
            os.mkdir(tray / 'part_images' / str(part_number))
        except FileExistsError:
            pass

    image_filenames = list((tray / 'tray_images').glob('*'))
    n_images = len(image_filenames)

    n = 0
    for image_filename in image_filenames:
        image = Image.open(image_filename)
        for part_location in part_locations:
            left = part_location[1]
            right = part_location[2]
            top = part_location[3]
            bottom = part_location[4]
            image_cropped = image.crop((left, top, right, bottom))
            path = tray / 'part_images' / str(part_location[0]) / image_filename.name
            image_cropped.save(path)
        n += 1
        sys.stdout.write(f'\r{n}/{n_images}')
        sys.stdout.flush()

    print('\n')
