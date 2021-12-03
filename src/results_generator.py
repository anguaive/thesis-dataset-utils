from pathlib import Path
from utils import dpath, rpath, Template, find_tdef_templates
from matchers import create_matcher
import os

def collect_missing_results(matcher, templates, force=False):
    missing = []
    variants = matcher.get_variants()

    for variant in variants:
        if matcher.rcode:
            v = variant + '-' + matcher.rcode
        else:
            v = variant
        for t in templates:
            full_path = rpath / t.tray / t.part / t.id / v
            if force or not full_path.is_dir():
                missing.append((full_path, variant, t))

    return missing

def gen_results(job, force=False):
    for tdef in job.tdefs:
        templates = find_tdef_templates(tdef)

        for mdef in job.mdefs:
            matcher = create_matcher(mdef, tdef)

            missing = collect_missing_results(matcher, templates, force)

            for m in missing:
                full_path, variant, t = m
                matcher.run(full_path, variant, t)
