import os
import sys
from pathlib import Path
from enum import Enum

results_path = (Path(__file__).parents[1] / 'results').resolve()

def get_trays_path(debug=False):
    if not debug:
        return (Path(__file__).parents[1]  / 'dataset').resolve()
    else:
        return (Path(__file__).parents[1]  / 'debug_dataset').resolve()

class Result:
    def __init__(self, n, p, t, state, purity):
        self.n = n
        self.p = p
        self.t = t
        self.state = state
        self.purity = purity

class TemplateState(Enum):
    PRESENT = 1,
    MISSING = 2,
    UNCERTAIN = 3

    def __str__(self):
        if self is TemplateState.PRESENT:
            s = 'present'
        elif self is TemplateState.MISSING:
            s = 'missing'
        else:
            s = 'uncertain'

        return s.rjust(9)

class TemplatePurity(Enum):
    CLEAN = 1,
    DIRTY = 2

    def __str__(self):
        if self is TemplatePurity.CLEAN:
            return 'clean'
        else:
            return 'dirty'

class Image():
    def __init__(self, path, pixmap=None):
        self.path = path
        self.pixmap = pixmap

    def __lt__(self, other):
        return self.path.name < other.path.name

class Template(Image):
    def __init__(self, path, pixmap=None):
        super().__init__(path, pixmap)
        self.purity = TemplatePurity[path.parts[-2].upper()]
        self.state = TemplateState[path.parts[-3].upper()]

    def load_image(self, scaling, grayscale=False):
        import cv2 as cv
        if grayscale:
            mode = cv.IMREAD_GRAYSCALE
        else:
            mode = cv.IMREAD_UNCHANGED
        image = cv.imread(str(self.path), mode)
        self.pixmap = cv.resize(image, dsize=None, fx=scaling, fy=scaling, interpolation=cv.INTER_AREA)

    def __str__(self):
        return f'{self.path.stem} {str(self.purity)} {str(self.state)}'

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

def read_cv_images(path, scaling, text='images'):
    print(f'Reading and downsampling {text}...')
    import cv2 as cv 
    images = [Image(p) for p in path.glob('*')]
    n_images = len(images)
    progress = 0

    for image in images:
        pixmap = cv.imread(str(path / image.path.name), cv.IMREAD_UNCHANGED)
        image.pixmap = cv.resize(pixmap, dsize=None, fx=scaling, fy=scaling, interpolation=cv.INTER_AREA)

        progress += 1
        sys.stdout.write(f'\r{progress}/{n_images}')
        sys.stdout.flush()
    print('\n')

    return images, n_images

def find_all_duplicates_in_folder(path, scaling, threshold, verbose=False):
    images, n_images = read_cv_images(path, scaling)

    print('Collecting duplicates...')
    progress = 0
    dupes = set([])
    while len(images) > 0:
        current = images.pop()
        current_dupes = find_duplicates_of_image(current, images, threshold, verbose)
        images = [i for i in images if i.path.name not in current_dupes]
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
        result = cv.matchTemplate(image.pixmap, cand.pixmap, cv.TM_CCOEFF_NORMED)
        if result > threshold:
            dupes.add(cand.path.name)

        if verbose:
            print(f'\tf({image.path.name}, {cand.path.name}) -> {result[0][0]:.2f}; { "identical" if result > threshold else "distinct"}')

    if verbose:
        print(f'Found {len(dupes)} duplicates for {image.path.name}\n')

    return dupes
