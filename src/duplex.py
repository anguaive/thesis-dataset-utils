from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from utils import TemplateState, rpath, epath, get_encoded_variant_name, read_results, Color, Hatching, collect_all_tray_templates, Result
from algorithms import TM
import os

def get_duplex_pairs(templates):
    pairs = {}
    for t in templates:
        if t.state is not TemplateState.UNCERTAIN:
            if (t.tray, t.part) not in pairs:
                pairs[(t.tray, t.part)] = [t]
            else:
                pairs[(t.tray, t.part)].append(t)
                
    filtered_pairs = {}
    for k,v in pairs.items():
        if len(v) != 2:
            continue

        if v[0].state is v[1].state:
            continue

        if v[0].state is TemplateState.MISSING:
            v[0],v[1] = v[1],v[0]

        filtered_pairs[k] = v

    return filtered_pairs

def gen_duplex_evals(alg, pairs):
    print('Generating duplex evaluations')
    variants = alg.get_variants()

    for (tray, part), [present, missing] in pairs.items():
        for variant in variants:
            is_ccoeff_normed = variant == 'TM_CCOEFF_NORMED'
            is_sqdiff = variant in ['TM_SQDIFF', 'TM_SQDIFF_NORMED']
            is_normed = variant in TM.normed_methods
            v = get_encoded_variant_name(variant, alg.code)
            path = epath / tray / 'duplex' / part / f'{present.id}-{missing.id}' / v
            os.makedirs(path, exist_ok=True)
            samples = collect_all_tray_templates(tray, present.part)

            present_results_file = rpath / tray / part / present.id / v / 'results.txt'
            missing_results_file = rpath / tray / part / missing.id / v / 'results.txt'
            present_results = read_results(samples, variant, present, present_results_file)
            present_results.sort(key=lambda res:res.id)
            missing_results = read_results(samples, variant, missing, missing_results_file)
            missing_results.sort(key=lambda res:res.id)

            sample_size = len(present_results)
            width = 0.28

            correctness = []
            for i in range(sample_size):
                p = present_results[i]
                m = missing_results[i]
                if (p.score > m.score and p.correct) or (p.score < m.score and m.correct):
                    correctness.append(True if not is_sqdiff else False)
                else:
                    correctness.append(False if not is_sqdiff else True)

            x = np.arange(0, sample_size)[:50]
            p_score = [p.score for p in present_results]
            m_score = [m.score for m in missing_results]
            p_colors = [Color.BLUE.value if c else Color.RED.value for c in correctness]
            m_colors = [Color.GREEN.value if c else Color.RED.value for c in correctness]
            clean = [p.clean for p in present_results]
            hatching = [Hatching.NONE.value if c else Hatching.DOTTED.value for c in clean]

            fig, ax = plt.subplots(figsize=(12,8), tight_layout=True)

            ax.bar(x-width/2, p_score[:50], width, edgecolor='k', linewidth=0.7, hatch=hatching[:50], color=p_colors[:50])
            ax.bar(x+width/2, m_score[:50], width, edgecolor='k', linewidth=0.7, hatch=hatching[:50], color=m_colors[:50])

            ax.set_xlabel('Sample')
            ax.set_ylabel('Score')
            ax.set_xticks([])

            if is_normed:
                ax.set_ylim((0, 1))
                ax.hlines(np.linspace(0, 1, 11), 0, 1, color='#dddddd',
                        transform=ax.get_yaxis_transform(), zorder=1,
                        linewidth=0.7)
            else:
                ax.set_ylim((0, 1.1 * max(max(p_score), max(m_score))))

            misscount = 0
            for s in correctness:
                if not s:
                    misscount += 1

            accuracy = (sample_size - misscount) / sample_size
            avg_time = (sum([res.elapsed for res in present_results]) + sum([res.elapsed for res in missing_results])) / sample_size

            handles = [
                    plt.Rectangle((0,0),1,1, color=Color.BLUE.value),
                    plt.Rectangle((0,0),1,1, color=Color.GREEN.value),
                    plt.Rectangle((0,0),1,1, color=Color.RED.value),
                    plt.Rectangle((0,0),1,1, fill=None, hatch=Hatching.DOTTED.value)
            ]
            labels = ['score (present)', 'score (missing)', 'misclassification', 'dirty']
            plt.legend(handles, labels)

            title = f'{variant} (duplex)\n{alg.params_str}\n{present.tray}/{present.part}/{present.id}-{missing.id}'

            plt.title(title)
            fig.savefig(path / 'figure.png')
            plt.close(fig)

            with open(path / 'evaluation.txt', 'w') as f:
                f.write(f'{variant}-duplex\n')
                f.write(f'{alg.params_str}\n')
                f.write(f'average time: {round(avg_time, 4)}\n')
                f.write(f'accuracy: {round(accuracy, 4)}\n')
                f.write(f'misses: {misscount}\n')
                f.write(f'sample size: {sample_size}\n')

def gen_duplex_templ_aggrs(pairs):
    print('Generating template level duplex aggregates')

    for (tray, part), [present, missing] in pairs.items():
        path = epath / tray / 'duplex' / part / f'{present.id}-{missing.id}' 

        algs = []
        evals = [f for f in path.rglob('*') if f.name == 'evaluation.txt']
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

        ticklabels = [f'{a[0]}\n{a[1]}' for a in algs]
        accs = [float(a[2]['accuracy']) for a in algs]
        times = [float(a[2]['average time']) for a in algs]
        width = 0.24

        fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)
        ax2 = ax.twinx()

        bar_accuracy = ax.bar(x - width / 2, accs, width, label='accuracy', color=Color.GREEN.value)
        ax.set_ylabel('Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(ticklabels, rotation='75', linespacing=0.5)
        ax.bar_label(bar_accuracy, padding=3, fmt='%.2f')

        bar_avg_time = ax2.bar(x + width / 2, times, width, label='average time', color=Color.BLUE.value)
        ax2.set_ylim(0, 1.1 * max(times))
        ax2.set_ylabel('Average time (ms)')
        ax2.bar_label(bar_avg_time, padding=3, fmt='%.2f')

        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2)

        ax.set_title(f'Comparison of duplex algorithms on {tray}/{part}/{present.id}-{missing.id}')

        try:
            os.mkdir(path / '_aggregates')
        except FileExistsError:
            pass

        fig.savefig(path / '_aggregates' / 'figure.png')
        plt.close(fig)
