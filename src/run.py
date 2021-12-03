#!/usr/bin/python3

from pathlib import Path
from job_parser import parse_job
from results_generator import gen_results
import argparse

parser = argparse.ArgumentParser(description='run')

parser.add_argument(
        'job',
        type=str,
        help='Path to job file')

parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Overwrite existing results')

args = parser.parse_args()

job = parse_job(args.job)
if not job:
    exit(1)

gen_results(job, args.force)


