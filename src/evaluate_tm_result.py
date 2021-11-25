#!/usr/bin/python3

from utils import results_path, Result
from matchers import tm_methods
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import operator
import argparse

parser = argparse.ArgumentParser(description='Evaluate TM results. Create a plot and a textfile detailing the findings.')

parser.add_argument(
        '-o', '--overwrite',
        action='store_true',
        help='Overwrite existing evaluations'
        )

parser.add_argument(
        '-c', '--count',
        type=int,
        help='How many evaluations to make before stopping'
        )

args = parser.parse_args()

cnt=0
COLORS={True: '#7484ec', False: '#dc8686'}
HATCHING={True: '', False: '.....'}

all_children = results_path.rglob('*')
results_to_evaluate = [p for p in all_children if p.is_dir()
        and p.name in tm_methods]

for result in results_to_evaluate:
    if not args.overwrite and (result / 'evaluation.txt').is_file():
        continue

    if args.count:
        if cnt == args.count:
            exit(0)
        cnt += 1

    print(f'Evaluating {result}...')

    is_thresholded = result.name.endswith('_NORMED')
    is_sqdiff = result.name.startswith('TM_SQDIFF')
    is_ccoeff_normed = result.name == 'TM_CCOEFF_NORMED'

    results = []
    with open(result / 'results.txt', 'r') as f:
        first_line_words = next(f).split()
        template_name = first_line_words[0]
        template_state = first_line_words[1]
        template_purity = first_line_words[2]

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

    # filter out results from input images where the presence of the part is
    # uncertain
    original_size = len(results)
    results = [res for res in results if res.state != 'uncertain']
    size = len(results)

    x = np.arange(0, size)
    y = [res.p for res in results]
    state = [True if res.state == template_state else False for res in results]
    purity = [True if res.purity == 'clean' else False for res in results]

    df = pd.DataFrame({'x': x, 'y': y, 'state': state, 'purity': purity})
    df = df.sort_values('y')

    colors=[COLORS[st] for st in df['state']]
    hatching=[HATCHING[pr] for pr in df['purity']]

    fig, ax = plt.subplots()
    ax.bar(x, df['y'], width=1, edgecolor='k', linewidth=0.7,
           color=colors, hatch=hatching)

    ax.set_xlabel('Tray image')
    ax.set_ylabel('Score')
    ax.set_xticks([])

    if is_thresholded:
        # Calculate Gini's impurity at each threshold location
        ginis = []
        for i in x[1:]:
            pl1 = df['state'][:i].value_counts().get(True, 0) / i
            pl2 = 1 - pl1
            pr1 = df['state'][i:].value_counts().get(True, 0) / (size-i)
            pr2 = 1 - pr1
            gini = (i / size) * 2 * pl1 * pl2 + ((size-i) / size) * 2 * pr1 * pr2
            ginis.append((i, gini))

        # Optimal threshold location is where the impurity is lowest
        # Could've used the miscount at each threshold location instead for basically the same result
        ginis.sort(key=operator.itemgetter(1))
        best_gini = ginis[0][0]
        threshold = (df['y'].iloc[best_gini-1] + df['y'].iloc[best_gini]) / 2

        misses = []
        for i, row in enumerate(df.to_numpy()):
            condition = (i+1 > best_gini) if is_sqdiff else (i < best_gini)
            if condition:
                if row[2]:
                    misses.append((i, row[0]))
            else:
                if not row[2]:
                    misses.append((i, row[0]))

        ax.set_ylim((0, 1))
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.hlines(np.linspace(0, 1, 11), 0, 1, color='#dddddd',
                transform=ax.get_yaxis_transform(), zorder=1,
                linewidth=0.7)
        plt.axhline(threshold, linewidth=0.7, color='k', dashes=(4, 2),
                zorder=-1)
        plt.text(0, threshold, 'threshold:' + str(round(threshold, 4)), 
                backgroundcolor='w')

    else:
        ax.set_ylim((0, max(y)))

    plt.title(result.name)
    fig.set_size_inches(12, 8)
    fig.savefig(result / 'figure.png', dpi=120)
    plt.close(fig)

    misrate = len(misses) / size
    avg_time = sum([res.t for res in results]) / size

    with open(result / 'evaluation.txt', 'w') as f:
        f.write(f'misrate: {round(misrate, 6)}\n')
        f.write(f'average_time: {round(avg_time, 6)}\n')
        if is_thresholded:
            f.write(f'optimal_threshold: {round(threshold, 4)}\n')
