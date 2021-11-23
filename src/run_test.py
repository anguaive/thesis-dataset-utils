#!/usr/bin/python3

import os
import argparse
from pathlib import Path
from utils import get_trays_path, results_path, Template, read_cv_images
from matchers import tm, tm_methods

class Job:
    def __init__(self, part, part_image, scaling, roi):
        self.part = part
        self.part_image = part_image
        self.scaling = scaling
        self.roi = roi

parser = argparse.ArgumentParser(description='...')

parser.add_argument(
        '-i', '--input_file',
        required=True,
        help='File that describes what jobs to run')

parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Use the debug dataset')

args = parser.parse_args()

input_file = Path(args.input_file)
if not input_file.is_file():
    print('not a file!')

jobs = []
with open(input_file, 'r') as f:
    method_name = next(f).strip()
    tray = next(f).strip()
    for line in f:
        words = line.split()
        part = words[0]
        part_image = words[1]
        scaling = words[2] if 2 < len(words) else 1.0
        roi = words[3] if 3 < len(words) else 20
        jobs.append(Job(part, part_image, scaling, roi))

method = eval(method_name)
trays_path = get_trays_path(args.debug)
tray_path = trays_path / tray

part_descriptors = {}
with open(tray_path / 'tray_descriptor.txt', 'r') as f:
    next(f)
    for line in f:
        part_location = line.split()
        part_descriptors[part_location[0]] = list(map(int, part_location[1:]))

tray_images, n_tray_images = read_cv_images(tray_path / 'tray_images',
        scaling=1, text='tray images')
tray_images.sort()

for job in jobs:
    print(f'Processing job {job.part} {job.part_image}')
    part_path = tray_path / 'part_images' / job.part
    if not part_path.is_dir():
        print(f'Part directory {part_path} does not exist! Skipping')
        continue
         
    templates = {}
    for p in part_path.rglob('*'):
        if p.is_file():
            templ = Template(p)
            templates[templ.path.stem] = templ
            
    template = templates.get(job.part_image, None)

    if not template:
        print(f'Part image {part_image} does not exist! Skipping')
        continue

    import cv2 as cv

    template.load_image(scaling, grayscale=True)

    left, right, top, bottom = part_descriptors[job.part]
    h = bottom - top + 2 * roi
    w = right - left + 2 * roi
    roi = round(job.roi / job.scaling)

    base_path = results_path / tray_path.name / part_path.name / job.part_image

    for method in tm_methods:
        path = base_path / method
        outline_path = path / 'outline_images'
        scoremap_path = path / 'scoremap_images'

        try:
            os.makedirs(path, exist_ok = True)
            os.mkdir(outline_path)
            os.mkdir(scoremap_path)
        except FileExistsError:
            pass

        with open(path / 'results.txt', 'w') as f:
            f.write(f'{template.path.stem} {template.state} {template.purity}\n')

    for tray_image in tray_images:
        current_name = tray_image.path.stem
        descriptor = templates[current_name]

        cropped_tray_pixmap = tray_image.pixmap[
                max(top-roi, 0):top-roi+h, 
                max(left-roi, 0):left-roi+w
                ]
        colored_input_pixmap = cv.resize(cropped_tray_pixmap, dsize=None, fx=job.scaling, fy=job.scaling, interpolation=cv.INTER_AREA)

        tm(base_path, colored_input_pixmap, current_name, descriptor, template)
