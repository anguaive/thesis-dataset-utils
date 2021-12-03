import os
import sys
from pathlib import Path
from enum import Enum
import math
import cv2 as cv

root_path = Path(__file__).parents[1]

dpath = root_path / 'dataset'
rpath = root_path / 'results'
epath = root_path / 'evaluations'

class Job:
    def __init__(self, name, mdefs, tdefs):
        self.name = name
        self.mdefs = mdefs
        self.tdefs = tdefs

    def __str__(self):
        name = f'Name: `{self.name}`\n'
        mdefs = 'Method definitions:\n'
        tdefs = 'Tray definitions:\n'
        for mdef in self.mdefs:
            mdefs += f'{mdef}\n'
        for tdef in self.tdefs:
            tdefs += f'{tdef}'

        return name + mdefs + tdefs

class MethodDefinition:
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def __str__(self):
        return f'  {self.name}: {self.params}'

class TrayDefinition:
    def __init__(self, name, pdefs):
        self.name = name
        self.pdefs = pdefs

    def __str__(self):
        name = f'  {self.name}'
        pdefs = ''
        for pdef in self.pdefs:
            pdefs += f'    {pdef}\n'

        return f'{name}:\n{pdefs}'

class PartDefinition:
    def __init__(self, name, ids):
        self.name = name
        self.ids = ids

    def __str__(self):
        name = f'{self.name}'
        ids = ''
        for id in self.ids:
            ids += id + ' '

        return f'{name}: {ids}'

class Result:
    def __init__(self, n, p, t, state, purity):
        self.n = n
        self.p = p
        self.t = t
        self.state = state
        self.purity = purity

class TemplateState(Enum):
    PRESENT = 'present'
    MISSING = 'missing'
    UNCERTAIN = 'uncertain' 

    def __str__(self):
        return self.value

class TemplatePurity(Enum):
    CLEAN = 'clean'
    DIRTY = 'dirty'

    def __str__(self):
        return self.value

class Image:
    def __init__(self, path):
        self.path = path

    def load(self, grayscale=False, scaling=None):
        if grayscale:
            mode = cv.IMREAD_GRAYSCALE
        else:
            mode = cv.IMREAD_UNCHANGED
        self.pixmap = cv.imread(str(self.path), mode)
        if scaling:
            self.rescale(scaling)

    def rescale(self, scaling):
        self.pixmap = cv.resize(self.pixmap, dsize=None, fx=scaling, fy=scaling,
                interpolation=cv.INTER_AREA)

    def __lt__(self, other):
        return self.path.name < other.path.name

class Template(Image):
    def __init__(self, tray, part, id):
        part_path = dpath / tray / 'part_images' / part
        paths = part_path.rglob('*')
        path = next((p for p in paths if p.stem == id), None)
        super().__init__(path)
        self.tray = tray
        self.part = part
        self.id = id
        self.purity = TemplatePurity(path.parts[-2])
        self.state = TemplateState(path.parts[-3])

    def __str__(self):
        return f'{self.tray}/{self.part}/{self.id} ({self.purity}, {self.state})'

def find_tdef_templates(tdef):
    templates = []
    for pdef in tdef.pdefs:
        for id in pdef.ids:
            templ = Template(tdef.name, pdef.name, id)
            templates.append(templ)

    return templates

def get_tray_n(tray):
    return len(list((dpath / tray / 'tray_images').glob('*')))

def get_tray_part_names(tray):
    return [p.stem for p in (dpath / tray / 'part_images').glob('*')]

def load_tray_descriptor(tray):
    desc = {}

    with open(dpath / tray / 'tray_descriptor.txt', 'r') as f:
        next(f)
        for line in f:
            part_location = line.split()
            desc[part_location[0]] = list(map(int, part_location[1:]))

    return desc

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
