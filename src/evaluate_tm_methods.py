#!/usr/bin/python3

from utils import results_path
from matchers import tm_methods
from matplotlib import pyplot as plt
import numpy as np
import argparse

COLORS={'blue': '#7484ec', 'red': '#dc8686', 'green': '#6ad780'}

parser = argparse.ArgumentParser(description='Evaluate TM methods. Create a plot detailing the findings.')

parser.add_argument(
        '-o', '--overwrite',
        action='store_true',
        help='Overwrite existing evaluations'
        )

parser.add_argument(
        '--show_one',
        action='store_true',
        help='Just show one please'
        )

args = parser.parse_args()

def is_figure_outdated(tm_folder_lms, figure_lm):
    for lm in tm_folder_lms:
        if lm > figure_lm:
            return True
    return False

class Evaluation:
    def __init__(self, misrate, avg_time, opt_threshold=None):
        self.misrate = misrate
        self.avg_time = avg_time
        self.opt_threshold = opt_threshold

    def __str__(self):
        return f'{self.misrate} {self.avg_time} {self.opt_threshold}'

all_children = results_path.rglob('*')
results_to_evaluate = [p for p in all_children if p.is_dir()
        and p.parents[2] == results_path]

for result in results_to_evaluate:
    figure = result / 'figure.png'
    tm_folders = [f for f in result.glob('*') if f.name in tm_methods]

    if not args.overwrite and figure.is_file():
        figure_lm = figure.stat().st_mtime
        tm_folder_lms = [f.stat().st_mtime for f in tm_folders]
        if not is_figure_outdated(tm_folder_lms, figure_lm):
            continue

    print(f'Evaluating {result}...')

    evals = {m: None for m in tm_methods}
    for tm_folder in tm_folders:
        if not (tm_folder / 'evaluation.txt').is_file():
            print(f'No evaluation in {tm_folder.name}! Skipping')
            continue

        is_normed = tm_folder.name.endswith('_NORMED')

        with open(tm_folder / 'evaluation.txt', 'r') as f:
            misrate = float(next(f).split()[1])
            avg_time = float(next(f).split()[1]) * 1000 # ms
            if is_normed:
                opt_threshold = float(next(f).split()[1])
            else:
                opt_threshold = 0
            evals[tm_folder.name] = Evaluation(misrate, avg_time, opt_threshold)

    x = np.arange(len(evals))
    width = 0.24
    misrates = [x.misrate for x in evals.values()]
    opt_thresholds = [x.opt_threshold for x in evals.values()]
    avg_times = [x.avg_time for x in evals.values()]

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    bar_misrate = ax.bar(x - width, misrates, width, label='misrate', color=COLORS['red'])
    bar_opt_threshold = ax.bar(x + width, opt_thresholds, width, label='opt_threshold', color=COLORS['blue'])
    ax.set_xlabel('TM Methods')
    ax.set_ylabel('Misrate / Optimal threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(tm_methods)
    ax.bar_label(bar_misrate, padding=3, fmt='%.2f')
    ax.bar_label(bar_opt_threshold, padding=3, fmt='%.2f')

    bar_avg_time = ax2.bar(x, avg_times, width, label='avg_time', color=COLORS['green'])
    ax2.set_ylim(0, max(avg_times) * 1.1)
    ax2.set_ylabel('Average time (ms)')
    ax2.bar_label(bar_avg_time, padding=3, fmt='%.2f')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    ax.set_title('Comparison of TM methods')

    if not args.show_one:
        fig.set_size_inches(12, 8)
        fig.savefig(result / 'methods_figure.png', dpi=120)
        plt.close(fig)
    else:
        plt.show()
        exit(0)


            





