#!/usr/bin/python3

import argparse
from pathlib import Path
from utils import results_path
from matchers import tm_methods
from matplotlib import pyplot as plt
import numpy as np

COLORS={'blue': '#7484ec', 'red': '#dc8686', 'green': '#6ad780'}

parser = argparse.ArgumentParser(description='Aggregate evals for multiple part - part_image pairs for a single tray')

parser.add_argument(
        '-i', '--input_file',
        required=True,
        help='Input file describing which evals to process'
        )

args = parser.parse_args()

input_file = Path(args.input_file)
if not input_file.is_file():
    print('not a file!')
    exit(1)

evals = []
with open(input_file, 'r') as f:
    next(f) # Method name not needed as of now
    tray = next(f).strip()
    for line in f:
        words = line.split()
        part = words[0]
        part_image = words[1]
        evals.append((part, part_image))

size = len(evals)

# Make sure there are no dupes
if size != len(set([e[0] for e in evals])):
    print('one or more template specified for the same part')
    exit(1)

final_method_evals = []
for method in tm_methods:
    total_misrate = 0
    total_avg_time = 0
    print(f'Method: {method}')
    for e in evals:
        efile = results_path / tray / e[0] / e[1] / method / 'evaluation.txt'
        if not efile.is_file():
            print('method eval file does not exist')
            exit(1)

        with open(efile, 'r') as ef:
            this_misrate = float(next(ef).split()[1])
            this_avg_time = float(next(ef).split()[1]) * 1000 # ms
            print(this_misrate)
            total_misrate += this_misrate
            total_avg_time += this_avg_time

    final_method_evals.append((total_misrate / size, total_avg_time))

x = np.arange(len(tm_methods))
width = 0.32
misrates = [fme[0] for fme in final_method_evals]
avg_times = [fme[1] for fme in final_method_evals]

fig, ax = plt.subplots()
ax2 = ax.twinx()

bar_misrate = ax.bar(x - width/2, misrates, width, label='misrate', color=COLORS['red'])
ax.set_xlabel('TM Methods')
ax.set_ylabel('Misrate')
ax.set_xticks(x)
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 11))
ax.set_xticklabels(tm_methods)
ax.bar_label(bar_misrate, padding=3, fmt='%.2f')

bar_avg_time = ax2.bar(x + width/2, avg_times, width, label='avg_time', color=COLORS['green'])
ax2.set_ylim(0, max(avg_times) * 1.1)
ax2.set_ylabel('Average time (ms)')
ax2.bar_label(bar_avg_time, padding=3, fmt='%.2f')

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

ax.set_title('Comparison of TM methods (aggregate)')
fig.set_size_inches(12, 8)
fig.savefig(results_path / tray / (Path(args.input_file).stem + '.png'))
plt.close(fig)
