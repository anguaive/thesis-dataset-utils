#!/usr/bin/python3

from pathlib import Path
from job_parser import parse_job
from utils import collect_tdef_templates, rpath, epath, TemplateState, get_encoded_variant_name
from algorithms import create_algorithm
from aggregates import gen_templ_aggrs, gen_job_aggrs
import argparse

def collect_missing_results(alg, templates, force=False):
    missing = []
    variants = alg.get_variants()

    for variant in variants:
        v = get_encoded_variant_name(variant, alg.rcode)

        for t in templates:
            full_path = rpath / t.tray / t.part / t.id / v
            if force or not full_path.is_dir():
                missing.append((full_path, variant, t))

    return missing

def collect_missing_evals(alg, templates, force=False):
    missing = []
    variants = alg.get_variants()

    for variant in variants:
        v = get_encoded_variant_name(variant, alg.ecode)

        for t in templates:
            if t.state is TemplateState.PRESENT:
                p = 'single_present'
            elif t.state is TemplateState.MISSING:
                p = 'single_missing'
            else:
                print(f'Skipping {t.tray}/{t.part}/{t.id} (state: `{t.state}`)')
                continue

            full_path = epath / t.tray / p / t.part / t.id / v
            if force or not full_path.is_dir():
                missing.append((full_path, variant, t))

    return missing

parser = argparse.ArgumentParser(description='run')

parser.add_argument(
        'job',
        type=str,
        help='Path to job file')

parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Overwrite existing results')

parser.add_argument(
        '--only_aggrs',
        action='store_true',
        help='Only redraw aggregates')

args = parser.parse_args()

job = parse_job(args.job)
if not job:
    exit(1)

all_job_templates = []
for tdef in job.tdefs:
    templates = collect_tdef_templates(tdef)
    all_job_templates += templates
    missing_templ_aggrs = set([])

    for mdef in job.mdefs:
        alg = create_algorithm(mdef, tdef)

        missing_results = collect_missing_results(alg, templates, args.force)
        for m in missing_results:
            if not args.only_aggrs:
                full_path, variant, t = m
                alg.run(full_path, variant, t)

        missing_evals = collect_missing_evals(alg, templates, args.force)
        for m in missing_evals:
            if not args.only_aggrs:
                full_path, variant, t = m
                alg.evaluate(full_path, variant, t)
        
        touched_templs = set([m[2] for m in missing_evals])
        missing_templ_aggrs |= touched_templs

    gen_templ_aggrs(missing_templ_aggrs)

gen_job_aggrs(job, all_job_templates)

