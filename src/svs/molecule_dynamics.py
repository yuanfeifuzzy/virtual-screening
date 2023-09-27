#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run molecule dynamics in an easy way
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import timedelta
from multiprocessing import Pool

import cmder
import utility
import pandas as pd
from rdkit import Chem

from svs import tools

logger = utility.setup_logger()

parser = argparse.ArgumentParser(prog='molecule-dynamics', description=__doc__.strip())
parser.add_argument('sdf', help="Path to a SDF file contains best docking pose for each cluster")
parser.add_argument('pdb', help="Path to a PDF file contains the structure for the docking target")
parser.add_argument('-o', '--outdir', default='.', type=str,
                    help="Path to a directory for saving output files, default: %(default)s")
parser.add_argument('-t', '--time', type=float, default=50, help="MD simulation time, default: %(default)s ns.")
parser.add_argument('-s', '--short', type=float, default=0, help="MD simulation time for short quick run, default: %(default)s ns.")
parser.add_argument('--rmsd', type=float, default=5.5, help="Max RMSD value for poses passing to long full run, default: %(default)s.")
parser.add_argument('-c', "--cpu", type=int, default=4, help="Number of CPUs can be used, default: %(default)s.")
parser.add_argument('-g', "--gpu", type=int, default=4, help="Number of GPUs can be used, default: %(default)s.")
parser.add_argument('-q', '--quiet', help='Process data quietly without debug and info message', action='store_true')
parser.add_argument('-v', '--verbose', help='Process data verbosely with debug and info message',
                    action='store_true')

parser.add_argument('--openmm_simulate', help='Path to openmm_simulate executable')

parser.add_argument('--wd', help="Path to work directory", default='.')
parser.add_argument('--task', type=int, default=0, help="ID associated with this task")
parser.add_argument('--submit', action='store_true', help="Submit job to job queue instead of directly running it")
parser.add_argument('--hold', action='store_true',
                    help="Hold the submission without actually submit the job to the queue")
parser.add_argument('--version', version=utility.get_version(__package__), action='version')

args = parser.parse_args()

utility.setup_logger(quiet=args.quiet, verbose=args.verbose)
outdir = utility.make_directory(args.outdir, task=args.task, status=-1)
output = outdir / 'md.rmsd.csv'
if output.exists():
    utility.debug_and_exit('MD results already exist, skip re-simulate', task=args.task, status=125)

tools.submit_or_skip(parser.prog, args, ['sdf', 'pdb'],
                     ['outdir', 'time', 'short', 'gpu', 'openmm_simulate', 'quiet', 'verbose', 'task'], day=21)

GPU_QUEUE = utility.gpu_queue(n=args.gpu)


def openmm_md(item):
    pose, target, name, tt = item
    out = f'{name}.trajectory.rmsd.csv'
    if Path(out).exists():
        return 0
    else:
        gpu_id = GPU_QUEUE.get()
        cmd = f'{args.openmm_simulate} {pose} {target} {name} --time {tt} --devnum {gpu_id} &> {name}.log'
        try:
            # p = cmder.run(cmd, env={'CUDA_VISIBLE_DEVICES': str(gpu_id)})
            p = cmder.run(cmd)
            if p.returncode:
                cmder.run(f'rm {pose}')
        finally:
            GPU_QUEUE.put(gpu_id)
        return p.returncode
    
    
def prepare_md(sdf, pdb, t):
    items = []
    with Chem.SDMolSupplier(str(sdf)) as f:
        for mol in f:
            name = mol.GetProp('_Name') or str(i)
            pose = f"{name}.sdf"
            Chem.SDWriter(pose).write(mol)
            items.append((pose, pdb, name, t))
    return items


def filter_rmsd(pdb='', t=0, max_rmsd=0.0):
    df, items, suffix = [], [], '.trajectory.rmsd.csv'
    for x in Path('.').glob(f'*{suffix}'):
        name = x.name.removesuffix(suffix)
        with open(x) as f:
            pose, rmsd = f.read().strip().split(',')
            if max_rmsd:
                if float(rmsd) < max_rmsd:
                    items.append((Path(pose).name, pdb, name, t))
                    cmder.run(f'rm {name}.trajectory.* {name}.log')
                else:
                    cmder.run(f'rm {name}.*')
            else:
                df.append({'pose': pose, 'docking_score': pose.rsplit('_', 1)[1].removesuffix('.sdf'), 'MD_RMSD': rmsd})
    return df, items


def main():
    sdf = utility.check_file(args.sdf, task=args.task, status=-1)
    pdb = utility.check_file(args.pdb, task=args.task, status=-1)
    utility.check_executable(args.openmm_simulate, task=args.task, status=-2)

    try:
        start = time.time()
        os.chdir(outdir)

        if Path('RMSD.csv').exists():
            utility.debug_and_exit(f'MD skipped since results already exist\n', task=args.task, status=125)
        else:
            if args.short:
                logger.debug(f'Running short quick MD with time set to {args.short} ns')
                items = prepare_md(sdf, pdb, args.short)
                utility.parallel_gpu_task(openmm_md, items)

                _, items = filter_rmsd(pdb=pdb, t=args.time, max_rmsd=args.rmsd)
                if len(items) > 100:
                    items = sorted(items, key=lambda x: float(x[0].removesuffix('.sdf').rsplit('_', 1)[1]))[:100]
                logger.debug(f'Running long full MD with time set to {args.time} ns')
                utility.parallel_gpu_task(openmm_md, items)
            elif args.time:
                logger.debug(f'Running single MD with time set to {args.time} ns')
                items = prepare_md(sdf, pdb, args.time)
                utility.parallel_gpu_task(openmm_md, items)
            else:
                utility.error_and_exit(f'Neither a single time nor short time was specified, MD cannot continue',
                                       task=args.task, status=-2)

            df, _ = filter_rmsd()
            if df:
                df = pd.DataFrame(df)
                df.to_csv('RMSD.csv', index=False)

            t = str(timedelta(seconds=time.time() - start))
            utility.debug_and_exit(f'MD complete in {t.split(".")[0]}\n', task=args.task, status=125)
    except Exception as e:
        utility.error_and_exit(f'MD failed due to\n{e}\n\n{traceback.format_exc()}', task=args.task, status=-1 * 125)


if __name__ == '__main__':
    main()
