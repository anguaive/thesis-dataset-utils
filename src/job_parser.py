import math
from utils import Job, MethodDefinition, TrayDefinition, PartDefinition, get_tray_n, get_tray_part_names

msep = '.m'
tsep = '.t'

class JobError(Exception):
    def __init__(self, err):
        self.err = err

    def __str__(self):
        return self.err

def _next(f):
    return next(f).rstrip() # to get rid of endlines

def parse_mdefs(f):
    mdefs = []
    
    try:
        _msep = _next(f)
        if _msep != msep:
            raise JobError('no msep or job_name isn\'t on one line')
        line = _next(f)
        while line != tsep: 
            words = line.split()
            method_name = words[0]
            method_params = eval('dict(' + ', '.join(words[1:]) + ')')
            mdef = MethodDefinition(method_name, method_params)
            mdefs.append(mdef)
            line = _next(f)
    except StopIteration:
        raise JobError('incomplete job')

    if len(mdefs) == 0:
        raise JobError('job contains no mdefs')

    return mdefs

def parse_tdefs(f):
    tdefs = []

    try:
        while True:
            pdefs = []

            line = _next(f)
            if line == tsep:
                raise JobError('tdef contains no tray name')
            else:
                tray_name = line
                n = get_tray_n(tray_name)
                valid_part_names = get_tray_part_names(tray_name)

            line = _next(f)
            while line != tsep:
                words = line.split()
                part_name = words[0]
                part_ids = words[1:]
                if part_name not in valid_part_names:
                    raise JobError(f'`{tray_name}/{part_name}` invalid part name')
                if len(part_ids) == 0:
                    raise JobError(f'`{tray_name}/{part_name}` contains no ids')
                for part_id in part_ids:
                    if int(part_id) > n:
                        raise JobError(f'{tray_name}/{part_name}/{part_id}` higher than tray depth')
                pdef = PartDefinition(part_name, part_ids)
                pdefs.append(pdef)
                line = _next(f)

            if len(pdefs) == 0:
                raise JobError('tdef `{tray_name}` contains no pdefs')

            tdef = TrayDefinition(tray_name, pdefs)
            tdefs.append(tdef)

    except StopIteration:
        tdef = TrayDefinition(tray_name, pdefs)
        tdefs.append(tdef)
        return tdefs

def parse_job(filename):
    with open(filename, 'r') as f:
        try:
            job_name = _next(f)
            mdefs = parse_mdefs(f)
            tdefs = parse_tdefs(f)
            return Job(job_name, mdefs, tdefs)
            
        except JobError as pje:
            print(pje)
            return None
