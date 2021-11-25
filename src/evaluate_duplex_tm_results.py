#!/usr/bin/python3

from matchers import tm_methods
from utils import results_path, Result
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import argparse

COLORS={'blue': '#7484ec', 'red': '#dc8686', 'green': '#6ad780'}
HATCHING={True: '', False: '.....'}

def read_duplex_results(base_path, template_names):
    final_results = {}
    for res in pairs[key]:
        method_folders = [f for f in (path / res).glob('*') if f.name in tm_methods]
        if len(method_folders) != 6:
            print(f'Results for {res} are incomplete! Skipping')
            exit(1)

        template_name = None
        template_state = None

        method_results = {}
        for folder in method_folders:
            results_file = folder / 'results.txt'
            is_ccoeff_normed = folder.name == 'TM_CCOEFF_NORMED'
            if not results_file.is_file():
                print(f'results.txt inside {folder} is missing! Skipping')
                exit(1)

            results = []
            with open(results_file, 'r') as f:
                first_line = next(f)
                if not template_name:
                    template_name = first_line.split()[0]
                    template_state = first_line.split()[1]
                for line in f:
                    words = line.split()

                    score = float(words[1])
                    if is_ccoeff_normed: # [-1, 1] -> [0, 1]
                        score = (score + 1) / 2

                    results.append(Result(
                        int(words[0]),
                        score,
                        float(words[2]),
                        words[3],
                        words[4]
                        ))

            results = [res for res in results if res.state != 'uncertain']
            results = results[:40]
            size = len(results)

            method_results[folder.name] = results

        final_results[template_name] = {'state': template_state, 'results': method_results}

    return final_results

def evaluate_method(base_path, prefix, method_name, p_results, m_results):

    is_sqdiff = method_name.startswith('TM_SQDIFF')
    is_thresholded = method_name.endswith('_NORMED')
    size = len(p_results)

    x = np.arange(0, size)
    py = [res.p for res in p_results]
    my = [res.p for res in m_results]
    ymax = []
    state = []
    for i in x:
        ymax.append(max(py[i], my[i]))
        actual = p_results[i].state
        diff = p_results[i].p - m_results[i].p
        if (actual == 'present' and diff > 0) or (actual == 'missing' and diff < 0):
            state.append(True if not is_sqdiff else False)
        else:
            state.append(False if not is_sqdiff else True)
    purity = [True if res.purity == 'clean' else False for res in
            p_results] # Doesn't matter which result set we're looking at, the state and purity are the same

    df = pd.DataFrame({'x': x, 'py': py, 'my': my, 'ymax': ymax, 'state': state, 'purity': purity})
    df = df.sort_values('ymax')

    py_colors=[COLORS['blue'] if st else COLORS['red'] for st in df['state']]
    my_colors=[COLORS['green'] if st else COLORS['red'] for st in df['state']]
    hatching = [HATCHING[pr] for pr in df['purity']]

    width=0.28

    fig, ax = plt.subplots()
    ax.bar(x-width/2, df['py'], width, edgecolor='k', linewidth=0.7, hatch=hatching, color=py_colors)
    ax.bar(x+width/2, df['my'], width, edgecolor='k', linewidth=0.7, hatch=hatching, color=my_colors)

    ax.set_xlabel('Tray image')
    ax.set_ylabel('Score')

    if is_thresholded:
        ax.set_ylim((0, 1))
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.hlines(np.linspace(0, 1, 11), 0, 1, color='#dddddd',
                transform=ax.get_yaxis_transform(), zorder=1,
                linewidth=0.7)
    else:
        ax.set_ylim((0, 1.1 * df['ymax'].iloc[-1]))

    miscount = 0
    for s in state:
        if not s:
            miscount += 1

    misrate = miscount / size
    avg_time = (sum([res.t for res in p_results]) + sum([res.t for res in m_results])) / size

    plt.title(method_name)
    fig.set_size_inches(12, 8)
    fig.savefig(base_path / f'duplex_{prefix}_{method_name}.png', dpi=120)
    plt.close(fig)

    with open(base_path / f'duplex_{prefix}_{method_name}.txt', 'w') as f:
        f.write(f'misrate: {round(misrate, 6)}\n')
        f.write(f'average_time: {round(avg_time, 6)}\n')
        

parser = argparse.ArgumentParser(description='Evaluate results of duplex TM')

parser.add_argument(
        '-i', '--input_file',
        required=True,
        help='File that describes the template pairs necessary for duplex evaluation. Same format as the job descriptor file')

parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Use the debug dataset')

args = parser.parse_args()

input_file = Path(args.input_file)
if not input_file.is_file():
    print('not a file!')

pairs = {}
with open(input_file, 'r') as f:
    next(f) # TM method is not needed for evaluation
    tray = next(f).strip()
    for line in f:
        words = line.split()
        part = words[0]
        part_image = words[1]
        # scaling and roi parameters aren't needed either
        if part in pairs:
            pairs[part].append(part_image)
        else:
            pairs[part] = [part_image]

for key in pairs.keys():
    path = results_path / tray / key

    if not path.is_dir():
        print(f'{path} does not exist! Skipping')
        continue

    if len(pairs[key]) != 2:
        print(f'Wrong number of jobs for part {key} (should be 2). Skipping')
        continue

    final_results = read_duplex_results(path, pairs[key])

    p_results = [r['results'] for r in final_results.values() if r['state'] == 'present']
    m_results = [r['results'] for r in final_results.values() if r['state'] == 'missing']

    if len(p_results) != 1 or len(m_results) != 1:
        print(f'There should be 1 missing and 1 present template. Skipping')
        continue

    prefix = '-'.join(list(final_results.keys()))

    p_results = p_results[0]
    m_results = m_results[0]

    for method in tm_methods:
        print(f'Evaluating {method} on {key} {list(final_results.keys())}')
        p_method_results = p_results[method]
        m_method_results = m_results[method]
        evaluate_method(path, prefix, method, p_method_results, m_method_results)
