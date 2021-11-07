#!/bin/python3

# Images are assumed to be of the same size and in .png format

import argparse
from pathlib import Path
from utils import confirm, trays_path, find_all_duplicates_in_folder, renumber
import os

parser = argparse.ArgumentParser(description="Deduplicate part images. Duplicates are discovered by the use of template matching with a (by default) very high threshold value. Matching can be ran on downsampled images to increase performance.")

parser.add_argument(
        '-t', '--threshold', 
        default=0.995, 
        type=float, 
        metavar='T',
        help='Value in the [0.0, 1.0] range. Two images are treated as identical if their similarity is above this value')
parser.add_argument(
        '-s', '--scaling', 
        default=1.0, 
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
args = parser.parse_args()

for tray in trays_path.glob('*'):
    if not (args.force or confirm(f'About to process tray: `{tray.name}`. Continue?')):
        continue

    print(f'Processing tray: `{tray.name}`.')
    for part_folder in (tray / 'part_images').glob('*'):
        print(f'Processing parts: `{part_folder.name}`.')
        dupes, n_images = find_all_duplicates_in_folder(part_folder, args.scaling, args.threshold, args.verbose)

        print(f'Found {len(dupes)} duplicates out of {n_images} images.')
        if len(dupes) == 0:
            continue

        if args.force or confirm(f'Found {len(dupes)} duplicates in {n_images} images. Are you sure you want to delete them?'):
            print('Deleting duplicates...')
            for dupe in dupes:
                os.remove(str(tray / 'part_images' / part_folder / dupe))

        if args.force or confirm(f'Do you wish to renumber the images?'):
            print('Renumbering images...')
            renumber(tray / 'part_images' / part_folder)
