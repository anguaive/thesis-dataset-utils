from pathlib import Path
from enum import Enum
from time import time
from utils import dpath, rpath, Image, load_tray_descriptor, collect_all_tray_templates, Color, Hatching, TemplateState, TemplatePurity, get_encoded_variant_name
import os
import cv2 as cv
import numpy as np
from operator import itemgetter
from matplotlib import pyplot as plt

common_params = set(['roi', 'scaling'])
tm_params = set(['duplex'])
fm_params = set(['nfeatures', 'sigma', 'minmatches', 'crosscheck'])

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
        }

abbrev_table_extended = abbrev_table.copy()
abbrev_table_extended['duplex'] = 'dp'
abbrev_table_extended['minmatches'] = 'min'

def get_param_encoding(config):
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

    return rcode, ecode

current_tray = None
tray_descriptor = {}

def create_algorithm(mdef, tdef):
    global tray_descriptor
    if tdef and tdef.name != current_tray:
        tray_descriptor = load_tray_descriptor(tdef.name)

    if mdef.name == 'tm':
        return TM(mdef, tdef)
    else:
        return FM(mdef, tdef)
    pass

class Result:
    def __init__(self, id, score, elapsed, state, expected_state, purity):
         self.id = id
         self.score = score
         self.elapsed = elapsed
         self.correct = state is expected_state
         self.clean = purity is TemplatePurity.CLEAN
         # print(f'{id} {score} {elapsed} {self.correct} {self.clean}')

    def __lt__(self, other):
        return self.score < other.score

class Algorithm:
    def __init__(self, mdef, tdef):
        self.roi = mdef.params.get('roi', 20)
        self.scaling = mdef.params.get('scaling', 1.0)
        self.params_str = mdef.params_str
        self.rcode, self.ecode = get_param_encoding(mdef.params)
        if tdef:
            self.load_tray(tdef.name)

    def get_variants(self):
        pass

    def load_tray(self, tray_name):
        self.tray_images = [Image(p) for p in (dpath / tray_name / 'tray_images').glob('*')]
        self.tray_images.sort()
        for i in self.tray_images:
            i.load(scaling=self.scaling)

    def run(self, full_path, variant, template):
        print(f'Matching: {variant} - {template.tray}/{template.part}/{template.id}')
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

    def evaluate(self, full_path, variant, template):
        print(f'Evaluating: {variant} - {template.tray}/{template.part}/{template.id}')
        os.makedirs(full_path, exist_ok=True)

        try:
            os.remove(full_path / 'evaluation.txt')
        except FileNotFoundError:
            pass

        self.samples = collect_all_tray_templates(template.tray, template.part)
        self.samples.sort()

        v = get_encoded_variant_name(variant, self.rcode)

        self.results_file = rpath / template.tray / template.part / template.id / v / 'results.txt'
        if not self.results_file.is_file():
            print(f'Results file {self.results_file} does not exist!')
            return None

class TM(Algorithm):
    methods = [
        'TM_SQDIFF', 
        'TM_CCORR', 
        'TM_CCOEFF', 
    ]
    normed_methods = [
        'TM_SQDIFF_NORMED',
        'TM_CCORR_NORMED', 
        'TM_CCOEFF_NORMED', 
    ]

    def __init__(self, mdef, tdef):
        super().__init__(mdef, tdef)
        self.duplex = mdef.params.get('duplex', False)

    def get_variants(self):
        return TM.methods + TM.normed_methods

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
                f.write(f'{stem} {val:.5f} {elapsed:.2f}\n')

    def evaluate(self, full_path, variant, template):
        super().evaluate(full_path, variant, template)
        results = []
        total_elapsed = 0
        is_normed = variant in TM.normed_methods
        is_sqdiff = variant in ['TM_SQDIFF', 'TM_SQDIFF_NORMED']
        is_ccoeff_normed = variant == 'TM_CCOEFF_NORMED'

        with open(self.results_file, 'r') as f:
            for line in f:
                words = line.split()
                id = words[0]

                state = next((s.state for s in self.samples if s.id == id), None)
                if state is TemplateState.UNCERTAIN:
                    continue

                purity = next((s.purity for s in self.samples if s.id == id), None)
                score = float(words[1])
                if is_ccoeff_normed:
                    score = (score + 1) / 2 # [-1, 1] -> [0, 1]
                elapsed = float(words[2])
                total_elapsed += elapsed
                result = Result(id, score, elapsed, state, template.state, purity)
                results.append(result)

        sample_size = len(results)
        results.sort()

        avg_time = total_elapsed / sample_size

        x = np.arange(0, sample_size)
        y = [res.score for res in results]
        states = [res.correct for res in results]
        colors = [Color.BLUE.value if res.correct else Color.RED.value for res in results]
        hatches = [Hatching.NONE.value if res.clean else Hatching.DOTTED.value for res in results]

        # TODO threshold

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x, y, width=1, edgecolor='k', linewidth=0.7, color=colors, hatch=hatches)
        ax.set_xlabel('Sample') 
        ax.set_ylabel('Score')
        ax.set_xticks([])

        if is_normed:
            ax.set_ylim((0, 1))
            ax.hlines(np.linspace(0, 1, 11), 0, 1, color='#dddddd',
                    transform=ax.get_yaxis_transform(), zorder=1,
                    linewidth=0.7)
        else:
            ax.set_ylim((0, 1.1 * max(y)))

        # Calculate Gini's impurity at each location
        ginis = []
        for i in x[1:]:
            pl1 = states[:i].count(True) / i
            pl2 = 1 - pl1
            pr1 = states[i:].count(True) / (sample_size-i)
            pr2 = 1 - pr1
            gini = (i / sample_size) * 2 * pl1 * pl2 + ((sample_size-i) / sample_size) * 2 * pr1 * pr2
            ginis.append((i, gini))

        # Optimal threshold location is where the impurity is lowest
        ginis.sort(key=itemgetter(1))
        best_gini = ginis[0][0]
        threshold = (y[best_gini-1] + y[best_gini]) / 2

        misses = []
        for i, res in enumerate(results):
            condition = (i+1 > best_gini) if is_sqdiff else (i < best_gini)
            if condition:
                if res.correct:
                    misses.append((i, res.id))
            else:
                if not res.correct:
                    misses.append((i, res.id))
        
        accuracy = (sample_size - len(misses)) / sample_size

        plt.axhline(threshold, linewidth=0.7, color='k', dashes=(4, 2),
                zorder=-1)
        plt.text(0, threshold, 'threshold:' + str(round(threshold, 4)), 
                backgroundcolor='w')

        handles = [
                plt.Rectangle((0,0),1,1, color=Color.BLUE.value),
                plt.Rectangle((0,0),1,1, color=Color.RED.value),
                plt.Rectangle((0,0),1,1, hatch=Hatching.DOTTED.value)
        ]
        labels = ['same state as template', 'different state than template', 'dirty']
        plt.legend(handles, labels)

        title = f'{variant}\n{self.params_str}\n{template.tray}/{template.part}/{template.id}'

        plt.title(title)
        fig.savefig(full_path / 'figure.png')
        plt.close(fig)

        with open(full_path / 'evaluation.txt', 'w') as f:
            f.write(f'{variant}\n')
            f.write(f'{self.params_str}\n')
            f.write(f'average time: {round(avg_time, 4)}\n')
            f.write(f'accuracy: {round(accuracy, 4)}\n')
            f.write(f'misses: {len(misses)}\n')
            f.write(f'sample size: {sample_size}\n')
            f.write(f'optimal threshold: {round(threshold, 4)}\n')

class FM(Algorithm):
    def __init__(self, mdef, tdef):
        super().__init__(mdef, tdef)
        self.detector = mdef.name.upper()
        # TODO params
        if self.detector == 'sift':
            pass
        elif self.detector == 'surf':
            pass
        elif self.detector == 'orb':
            pass
        elif self.detector == 'akaze':
            pass

    def get_variants(self):
        return [self.detector]

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
                f.write(f'{stem} {len(good_matches)} {elapsed:.2f}\n')

    def evaluate(self, full_path, variant, template):
        super().evaluate(full_path, variant, template)
        results = []
        total_elapsed = 0

        with open(self.results_file, 'r') as f:
            for line in f:
                words = line.split()
                id = words[0]

                state = next((s.state for s in self.samples if s.id == id), None)
                if state is TemplateState.UNCERTAIN:
                    continue

                purity = next((s.purity for s in self.samples if s.id == id), None)
                score = float(words[1])
                elapsed = float(words[2])
                total_elapsed += elapsed
                result = Result(id, score, elapsed, state, template.state, purity)
                results.append(result)

        sample_size = len(results)
        results.sort()

        avg_time = total_elapsed / sample_size

        x = np.arange(0, sample_size)
        y = [res.score for res in results]
        states = [res.correct for res in results]
        colors = [Color.BLUE.value if res.correct else Color.RED.value for res in results]
        hatches = [Hatching.NONE.value if res.clean else Hatching.DOTTED.value for res in results]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x, y, width=1, edgecolor='k', linewidth=0.7, color=colors, hatch=hatches)
        ax.set_xlabel('Sample') 
        ax.set_ylabel('Matched features')
        ax.set_xticks([])
        ax.set_ylim((0, max(1.1 * max(y), 10)))

        # Calculate Gini's impurity at each location
        ginis = []
        for i in x[1:]:
            pl1 = states[:i].count(True) / i
            pl2 = 1 - pl1
            pr1 = states[i:].count(True) / (sample_size-i)
            pr2 = 1 - pr1
            gini = (i / sample_size) * 2 * pl1 * pl2 + ((sample_size-i) / sample_size) * 2 * pr1 * pr2
            ginis.append((i, gini))

        # Optimal threshold location is where the impurity is lowest
        ginis.sort(key=itemgetter(1))
        best_gini = ginis[0][0]
        threshold = (y[best_gini-1] + y[best_gini]) / 2

        misses = []
        for i, res in enumerate(results):
            if i < best_gini:
                if res.correct:
                    misses.append((i, res.id))
            else:
                if not res.correct:
                    misses.append((i, res.id))
        
        accuracy = (sample_size - len(misses)) / sample_size

        plt.axhline(threshold, linewidth=0.7, color='k', dashes=(4, 2),
                zorder=-1)
        plt.text(0, threshold, 'threshold:' + str(round(threshold, 4)), 
                backgroundcolor='w')

        handles = [
                plt.Rectangle((0,0),1,1, color=Color.BLUE.value),
                plt.Rectangle((0,0),1,1, color=Color.RED.value),
                plt.Rectangle((0,0),1,1, hatch=Hatching.DOTTED.value)
        ]
        labels = ['same state as template', 'different state than template', 'dirty']
        plt.legend(handles, labels)

        title = f'{variant}\n{self.params_str}\n{template.tray}/{template.part}/{template.id}'

        plt.title(title)
        fig.savefig(full_path / 'figure.png')
        plt.close(fig)

        with open(full_path / 'evaluation.txt', 'w') as f:
            f.write(f'{variant}\n')
            f.write(f'{self.params_str}\n')
            f.write(f'average time: {round(avg_time, 4)}\n')
            f.write(f'accuracy: {round(accuracy, 4)}\n')
            f.write(f'misses: {len(misses)}\n')
            f.write(f'sample size: {sample_size}\n')
            f.write(f'optimal threshold: {round(threshold, 4)}\n')
