#!/bin/python3

# Images are assumed to be of the same size and in .png format

import argparse
from pathlib import Path
from utils import confirm, get_trays_path, find_all_duplicates_in_folder, renumber
import os

parser = argparse.ArgumentParser(description="Deduplicate tray imagse. Duplicates are discovered by the use of template matching with a (by default) very high threshold value. Matching can be ran on downsampled images to increase performance.")

parser.add_argument(
        '-t', '--threshold', 
        default=0.995, 
        type=float, 
        metavar='T',
        help='Value in the [0.0, 1.0] range. Two images are treated as identical if their similarity is above this value')

parser.add_argument(
        '-s', '--scaling', 
        default=0.2, 
        type=float, 
        metavar='S',
        help='Value in the [0.0, 1.0] range. Scale factor (along both axes)')

parser.add_argument(
        '-v', '--verbose', 
        action='store_true')

parser.add_argument(
        '-f', '--force', 
        action='store_true',
        help='Perform actions without asking for user confirmation')

parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Use the debug dataset')

args = parser.parse_args()

trays_path = get_trays_path(args.debug)

for tray in trays_path.glob('*'):
    if not (args.force or confirm(f'About to process tray: `{tray.name}`. Continue?')):
        continue

    print(f'Processing tray: `{tray.name}`.')
    dupes, n_images = find_all_duplicates_in_folder(tray / 'tray_images', args.scaling, args.threshold, args.verbose)

    print(f'Found {len(dupes)} duplicates out of {n_images} images.')
    if len(dupes) == 0:
        continue

    if args.force or confirm(f'Found {len(dupes)} duplicates in {n_images} images. Are you sure you want to delete them?'):
        print('Deleting duplicates...')
        for dupe in dupes:
            os.remove(str(tray / 'tray_images' / dupe))

    if args.force or confirm(f'Do you wish to renumber the images?'):
        print('Renumbering images...')
        renumber(tray / 'tray_images')
