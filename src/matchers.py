from pathlib import Path
from enum import Enum
from time import time
from utils import dpath, rpath, Image, load_tray_descriptor
import os
import cv2 as cv
import numpy as np

common_params = set(['roi', 'scaling'])
tm_params = set(['duplex'])
fm_params = set(['nfeatures', 'sigma', 'minmatches', 'crosscheck', 'flann'])

abbrev_table = {
    'roi': 'r',
    'scaling': 's',
    'nfeatures': 'nf',
    'nOctaveLayers': 'nol',
    'contrastThreshold': 'ct',
    'edgeThreshold': 'et',
    'sigma': 'sg',
    'scaleFactor': 'sf',
    'nlevels': 'nl',
    'firstLevel': 'fl',
    'WTA_K': 'w',
    #TODO
    'crosscheck': 'c',
    'flann': 'n'
        }

abbrev_table_extended = abbrev_table.copy()
abbrev_table_extended['duplex'] = 'dp'
abbrev_table_extended['minmatches'] = 'min'

class MatchingMethod(Enum):
    TM = 'tm'
    SIFT = 'sift'
    SURF = 'surf'
    ORB = 'orb'
    AKAZE = 'akaze'

    def __str__(self):
        return self.value
    
    def lookup_params(self):
        if self.value == 'tm':
            return common_params | tm_params
        else:
            return common_params | fm_params

current_tray = None
tray_descriptor = {}

def create_matcher(mdef, tdef):
    global tray_descriptor
    if tdef.name != current_tray:
        tray_descriptor = load_tray_descriptor(tdef.name)

    if mdef.name == 'tm':
        return TM(mdef.params, tdef.name)
    else:
        algorithm = MatchingMethod(mdef.name)
        return FM(algorithm, mdef.params, tdef.name)
    pass

class Matcher:
    def __init__(self, config, tray_name):
        self.roi = config.get('roi', 20)
        self.scaling = config.get('scaling', 1.0)
        self.load_param_encodings(config)
        self.load_tray(tray_name)

    def get_variants(self):
        pass

    def load_tray(self, tray_name):
        self.tray = tray_name
        self.tray_images = [Image(p) for p in (dpath / tray_name / 'tray_images').glob('*')]
        self.tray_images.sort()
        for i in self.tray_images:
            i.load(scaling=self.scaling)

    def load_param_encodings(self, config):
        paramlist = list(config.items())
        paramlist.sort(key=lambda x:x[0])

        rcode = ''
        ecode = ''
        for key, value in paramlist:
            abbrev = abbrev_table_extended[key]
            val = str(round(value, 2)).replace('.', ',')
            ecode += f'{abbrev}{val}'
            if key in abbrev_table:
                rcode += f'{abbrev}{val}'

        self.rcode = rcode
        self.ecode = ecode

    def run(self, full_path, variant, template):
        print(f'{variant} - {self.tray}/{template.part}/{template.id}')
        os.makedirs(full_path, exist_ok=True)

        try:
            os.remove(full_path / 'results.txt')
        except FileNotFoundError:
            pass

        left, right, top, bottom = tray_descriptor[template.part]
        left = round(left * self.scaling)
        right = round(right * self.scaling)
        top = round(top * self.scaling)
        bottom = round(bottom * self.scaling)
        h = bottom - top + 2 * self.roi
        w = right - left + 2 * self.roi
        self.tlwh = (top, left, w, h)

class TM(Matcher):
    methods = [
        'TM_SQDIFF', 
        'TM_SQDIFF_NORMED',
        'TM_CCORR', 
        'TM_CCORR_NORMED', 
        'TM_CCOEFF', 
        'TM_CCOEFF_NORMED', 
    ]

    def __init__(self, config, tray_name):
        super().__init__(config, tray_name)
        self.duplex = config.get('duplex', False)

    def get_variants(self):
        return TM.methods

    def run(self, full_path, variant, template):
        super().run(full_path, variant, template)
        outline_path = full_path / 'outline_images'
        scoremap_path = full_path / 'scoremap_images'
        try:
            os.mkdir(outline_path)
            os.mkdir(scoremap_path)
        except FileExistsError:
            pass

        template.load(grayscale=True, scaling=self.scaling)

        method = eval('cv.' + variant)
        tw, th = template.pixmap.shape[::-1]
        top, left, w, h = self.tlwh

        for tray_image in self.tray_images:
            stem = tray_image.path.stem

            cropped_pixmap = tray_image.pixmap[
                    max(top-self.roi, 0):top-self.roi+h,
                    max(left-self.roi, 0):left-self.roi+w
                    ]

            outline_pixmap = cropped_pixmap.copy()
            scene = cv.cvtColor(cropped_pixmap, cv.COLOR_RGB2GRAY)

            # print(variant)
            # print(template.path)
            # print('\ttray:' + str(tray_image.pixmap.shape))
            # print('\tscene:' + str(scene.shape))
            # print('\ttempl:' + str(template.pixmap.shape))
            # print('\t' + str(top), left, w, h, self.roi)

            t0 = time()
            scoremap = cv.matchTemplate(scene, template.pixmap, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(scoremap)
            t1 = time()
            elapsed = (t1 - t0) * 1000 # ms

            # SQDIFF measures distance -> low value <=> good match
            # other methods measure similarity -> high value <=> good match
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                val = min_val
                loc = min_loc
            else:
                val = max_val
                loc = max_loc

            loc_br = (loc[0] + tw, loc[1] + th)
            cv.rectangle(outline_pixmap, loc, loc_br, (0, 255, 0), 4)

            # The matrix returned by matchTemplate contains single-channel 32-bit
            # floats; we use minmax normalization so the highest value becomes 1
            # and the lowest becomes 0; then we sample these values into unsigned
            # 8-bit values so the matrix could be displayed as an image
            cv.normalize(scoremap, scoremap, 0, 1, cv.NORM_MINMAX)
            scoremap = cv.convertScaleAbs(scoremap, alpha=255)

            cv.imwrite(str(outline_path / stem) + '.png', outline_pixmap)
            cv.imwrite(str(scoremap_path / stem) + '.png', scoremap)
            with open(full_path / 'results.txt', 'a') as f:
                f.write(f'{int(stem)} {val:.5f} {elapsed:.2f}\n')


class FM(Matcher):
    def __init__(self, algorithm, config, tray_name):
        super().__init__(config, tray_name)
        self.algorithm = algorithm
        self.flann = config.get('flann', False)
        # TODO(params)
        if algorithm is MatchingMethod['SIFT']:
            pass
        elif algorithm is MatchingMethod['SURF']:
            pass
        elif algorithm is MatchingMethod['ORB']:
            pass
        elif algorithm is MatchingMethod['AKAZE']:
            pass

    def get_variants(self):
        return [self.algorithm.name]

    def run(self, full_path, variant, template):
        super().run(full_path, variant, template)
        correspondence_path = full_path / 'correspondence_images'
        try:
            os.mkdir(correspondence_path)
        except FileExistsError:
            pass

        template.load(grayscale=True, scaling=self.scaling)

        if variant == 'SURF':
            ft = eval('cv.xfeatures2d.SURF_create()')
        else:
            ft = eval(f'cv.{variant}_create()')

        top, left, w, h = self.tlwh

        for tray_image in self.tray_images:
            stem = tray_image.path.stem

            cropped_pixmap = tray_image.pixmap[
                    max(top-self.roi, 0):top-self.roi+h,
                    max(left-self.roi, 0):left-self.roi+w
                    ]

            correspondence_pixmap = cropped_pixmap.copy()
            scene = cv.cvtColor(cropped_pixmap, cv.COLOR_RGB2GRAY)

            t0 = time()
            templ_kp, templ_des = ft.detectAndCompute(template.pixmap, None)
            scene_kp, scene_des = ft.detectAndCompute(scene, None)

            good_matches = []
            # TODO: nicer way to find out whether its None or an ndarray
            if type(templ_des) is np.ndarray and type(scene_des) is np.ndarray:

                # TODO distance and crosscheck as params
                bf = cv.BFMatcher()
                matches = bf.knnMatch(templ_des, scene_des, 2)

                for pair in matches:
                    if len(pair) == 1:
                        good_matches.append(pair[0])
                    else:
                        x,y = pair
                        # TODO lowes ratio as param
                        if x.distance < y.distance * 0.9:
                            good_matches.append(x)

                good_matches = sorted(good_matches, key=lambda x:x.distance)

            t1 = time()
            elapsed = (t1 - t0) * 1000 # ms

            correspondence_image = cv.drawMatches(template.pixmap, templ_kp,
                    scene, scene_kp, good_matches[:10], None, (0, 255, 0))

            cv.imwrite(str(correspondence_path / stem) + '.png',
                    correspondence_image)
            with open(full_path / 'results.txt', 'a') as f:
                f.write(f'{int(stem)} {len(good_matches)} {elapsed:.2f}\n')
