import os
import sys
from pathlib import Path

trays_path = Path(os.path.dirname(os.path.realpath(__file__)) + '/../dataset').resolve()

class cvImage:
    def __init__(self, name, data=None):
        self.name = name
        self.data = data

def number_width(number):
    from math import log10
    return int(log10(number) + 1)

def confirm(question):
    while True:
        reply = str(input(question + ' (Y/n): ')).strip()[0].lower()
        if reply == 'y':
            return True
        elif reply == 'n':
            return False
        else:
            print('Invalid input. Try again.')
            continue

def renumber(path):
    n = 0
    filenames = [p for p in list(path.glob('*'))]
    filenames.sort()
    digits = number_width(len(filenames))
    for filename in filenames:
        os.rename(str(path / filename), str(path / f'{str(n).zfill(digits)}{filename.suffix}'))
        n += 1

def find_all_duplicates_in_folder(path, scaling, threshold, verbose=False):
    print('Reading and downsampling images...')
    import cv2 as cv
    images = [cvImage(p.name) for p in path.glob('*')]
    n_images = len(images)
    progress = 0

    for image in images:
        data = cv.imread(str(path / image.name), cv.IMREAD_UNCHANGED)
        image.data = cv.resize(data, dsize=None, fx=scaling, fy=scaling, interpolation=cv.INTER_AREA)

        progress += 1
        sys.stdout.write(f'\r{progress}/{n_images}')
        sys.stdout.flush()
    print('\n')

    print('Collecting duplicates...')
    progress = 0
    dupes = set([])
    while len(images) > 0:
        current = images.pop()
        current_dupes = find_duplicates_of_image(current, images, threshold, verbose)
        images = [i for i in images if i.name not in current_dupes]
        dupes |= current_dupes

        progress = n_images - len(images)
        sys.stdout.write(f'\r{progress}/{n_images}')
        sys.stdout.flush()
    print('\n')

    if verbose:
        print(f'Duplicates:')
        for dupe in dupes:
            print('\t' + dupe)

    return dupes, n_images

def find_duplicates_of_image(image, candidates, threshold, verbose=False):
    import cv2 as cv
    dupes = set([])
    for cand in candidates:
        result = cv.matchTemplate(image.data, cand.data, cv.TM_CCOEFF_NORMED)
        if result > threshold:
            dupes.add(cand.name)

        if verbose:
            print(f'\tf({source.name}, {templ.name}) -> {result[0][0]:.2f}; { "identical" if result > threshold else "distinct"}')

    if verbose:
        print(f'Found {len(dupes)} duplicates for {source.name}\n')

    return dupes
