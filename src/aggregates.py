from pathlib import Path
import os
from matplotlib import pyplot as plt
from utils import TemplateState, Color, rpath, epath, get_encoded_variant_name, collect_all_tray_templates
from algorithms import create_algorithm, Result
import math
import numpy as np

def gen_templ_aggrs(missing_templ_aggrs):
    for t in missing_templ_aggrs:
        if t.state is TemplateState('present'):
            p = 'single_present'
        elif t.state is TemplateState('missing'):
            p = 'single_missing'
        else:
            print(f'Template {t.tray}/{t.part}/{t.id} has the state {t.state} and cannot be used for evaluation')
            continue

        full_path = epath / t.tray / p / t.part / t.id

        algs = []
        evals = [f for f in full_path.rglob('*') if f.name == 'evaluation.txt']
        for e in evals:
            with open(e, 'r') as f:
                name = next(f)
                params_str = next(f)
                perf = {}
                for line in f:
                    parts = line.partition(':')
                    key = parts[0].strip()
                    value = parts[2].strip()
                    if len(key) and len(value):
                        perf[key] = value

            algs.append((name, params_str, perf))

        algs.sort(key=lambda x:x[0]) 
        x = np.arange(len(algs))  
        # ticklabels = []
        # for a in algs:
        #     tl = f'{a[0]}\n'
        #     for param in a[1].split():
        #         tl += f'{param}\n'
        #     ticklabels.append(tl)

        ticklabels = [a[0] for a in algs]
        accs = [float(a[2]['accuracy']) for a in algs]
        times = [float(a[2]['average time']) for a in algs]
        width = 0.24

        fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)
        ax2 = ax.twinx()

        bar_accuracy = ax.bar(x - width / 2, accs, width, label='accuracy', color=Color.GREEN.value)
        ax.set_ylabel('Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(ticklabels, rotation='45')
        ax.bar_label(bar_accuracy, padding=3, fmt='%.2f')

        bar_avg_time = ax2.bar(x + width / 2, times, width, label='average time', color=Color.BLUE.value)
        ax2.set_ylim(0, 1.1 * max(times))
        ax2.set_ylabel('Average time (ms)')
        ax2.bar_label(bar_avg_time, padding=3, fmt='%.2f')

        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2)

        ax.set_title(f'Comparison of algorithms on {t.tray}/{t.part}/{t.id}')

        try:
            os.mkdir(full_path / '_aggregates')
        except FileExistsError:
            pass

        fig.savefig(full_path / '_aggregates' / 'figure.png')
        plt.close(fig)

def gen_job_aggr_scatter(path, algdef, templates):
    templates = list(templates) # dict_values -> list
    encoded = get_encoded_variant_name(algdef[0], algdef[1])

    total = len(templates)
    cols = min(total, 6)
    rows = math.ceil(total / cols)
    
    fig = plt.figure(figsize=(12,8), tight_layout=True)
    for i in range(total):
        t = templates[i]
        samples = collect_all_tray_templates(t.tray, t.part)
        file = rpath / t.tray / t.part / t.id / encoded / 'results.txt'
        results = []
        with open(file, 'r') as f:
            for line in f:
                words = line.split()
                id = words[0]

                state = next((s.state for s in samples if s.id == id), None)
                if state is TemplateState.UNCERTAIN:
                    continue

                purity = next((s.purity for s in samples if s.id == id), None)
                score = float(words[1])
                elapsed = float(words[2])
                result = Result(id, score, elapsed, state, t.state, purity)
                results.append(result)

        sample_size = len(results)
        results.sort()

        x = np.arange(0, sample_size)
        y = [res.score for res in results]
        colors = [Color.BLUE.value if res.correct else Color.RED.value for res in results]

        ax = fig.add_subplot(rows,cols,i+1)
        ax.scatter(x, y, color=colors)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim((0, max(1.1 * max(y), 1)))
        plt.title(f'{t.tray}/{t.part}', fontsize=10)
        plt.suptitle(encoded)

    save_name = get_encoded_variant_name(algdef[0], algdef[2])
    fig.savefig(path / f'{save_name}.png')
    plt.close(fig)

def gen_job_aggr(path, algdefs, templates, title):
    os.makedirs(path, exist_ok=True)

    algs = []
    for algdef in algdefs:
        gen_job_aggr_scatter(path, algdef, templates)

        encoded = get_encoded_variant_name(algdef[0], algdef[2])

        total_time = 0
        total_misses = 0
        total_sample_size = 0
        for t in templates:
            if t.state is TemplateState.PRESENT:
                p = 'single_present'
            elif t.state is TemplateState.MISSING:
                p = 'single_missing'
            else:
                print(f'Skipping {t.tray}/{t.part}/{t.id} (state: `{t.state}`)')
                continue

            e = epath / t.tray / p / t.part / t.id / encoded / 'evaluation.txt'
                
            with open(e, 'r') as f:
                name = next(f)
                params_str = next(f)
                perf = {}
                for line in f:
                    parts = line.partition(':')
                    key = parts[0].strip()
                    value = parts[2].strip()
                    if len(key) and len(value):
                        perf[key] = value

                total_time += float(perf['average time'])
                total_misses = int(perf['misses'])
                total_sample_size = int(perf['sample size'])

        total_acc = (total_sample_size - total_misses) / total_sample_size
        algs.append((name, params_str, total_acc, total_time))

    algs.sort(key=lambda x:x[0])
    x = np.arange(len(algs))
    ticklabels = [a[0] for a in algs]
    accs = [a[2] for a in algs]
    times = [a[3] for a in algs]
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)
    ax2 = ax.twinx()

    bar_accuracy = ax.bar(x - width / 2, accs, width, label='accuracy', color=Color.GREEN.value)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels, rotation='45')
    ax.bar_label(bar_accuracy, padding=3, fmt='%.2f')

    bar_avg_time = ax2.bar(x + width / 2, times, width, label='average time', color=Color.BLUE.value)
    ax2.set_ylim(0, 1.1 * max(times))
    ax2.set_ylabel('Average time (ms)')
    ax2.bar_label(bar_avg_time, padding=3, fmt='%.2f')

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2)

    ax.set_title(f'Comparison of algorithms (aggregate, {title})')

    fig.savefig(path / 'figure.png')
    plt.close(fig)
    
def gen_job_aggrs(job, templates):
    os.makedirs(epath / '_aggregates' / job.name, exist_ok=True)

    filtered = {}
    for t in templates:
        if (t.tray, t.part) not in filtered:
            if t.state is not TemplateState.UNCERTAIN:
                filtered[(t.tray, t.part)] = t

    algdefs = []
    for mdef in job.mdefs:
        alg = create_algorithm(mdef, None)
        for v in alg.get_variants():
            algdefs.append((v, alg.rcode, alg.ecode))

    gen_job_aggr(epath / '_aggregates' / job.name, algdefs, filtered.values(), 'all trays')

    for tdef in job.tdefs:
        tdef_filtered = [v for (k, v) in filtered.items() if k[0] == tdef.name]
        gen_job_aggr(epath / tdef.name / '_aggregates' / job.name, algdefs, tdef_filtered, tdef.name)
