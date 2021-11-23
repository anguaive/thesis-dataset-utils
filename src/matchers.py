from pathlib import Path
from time import time

tm_methods = [
        'TM_SQDIFF', 
        'TM_SQDIFF_NORMED',
        'TM_CCORR', 
        'TM_CCORR_NORMED', 
        'TM_CCOEFF', 
        'TM_CCOEFF_NORMED', 
        ]

def tm(base_path, original_input_pixmap, current_name, input_descriptor, template):
    import cv2 as cv

    tw, th = template.pixmap.shape[::-1]

    for method in tm_methods:
        colored_input_pixmap = original_input_pixmap.copy()
        input_pixmap = cv.cvtColor(colored_input_pixmap, cv.COLOR_RGB2GRAY)
        method_id = eval('cv.' + method)
        path = base_path / method
        outline_path = path / 'outline_images'
        scoremap_path = path / 'scoremap_images'

        t0 = time()
        scoremap = cv.matchTemplate(input_pixmap, template.pixmap, method_id)
        t1 = time()
        elapsed = t1 - t0

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(scoremap)
        # SQDIFF measures distance -> low value <=> good match
        # other methods measure similarity -> high value <=> good match
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF]:
            val = min_val
            loc = min_loc
        else:
            val = max_val
            loc = max_loc

        loc_br = (loc[0] + tw, loc[1] + th)
        cv.rectangle(colored_input_pixmap, loc, loc_br, (0, 255, 0), 4)

        # The matrix returned by matchTemplate contains single-channel 32-bit
        # floats; we use minmax normalization so the highest value becomes 1
        # and the lowest becomes 0; then we sample these values into unsigned
        # 8-bit values so the matrix could be displayed as an image
        cv.normalize(scoremap, scoremap, 0, 1, cv.NORM_MINMAX)
        scoremap = cv.convertScaleAbs(scoremap, alpha=255)

        cv.imwrite(str(outline_path / current_name) + '.png', colored_input_pixmap)
        cv.imwrite(str(scoremap_path / current_name) + '.png', scoremap)
        with open(path / 'results.txt', 'a') as f:
            f.write(f'{int(current_name)} {val:.5f} {elapsed:.5f} {input_descriptor.state} {input_descriptor.purity}\n')


def fm(tray, part, part_image, param1, param2, debug=False):
    pass
