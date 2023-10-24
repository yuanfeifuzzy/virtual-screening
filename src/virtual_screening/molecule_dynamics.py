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
from multiprocessing import Pool, Queue

import cmder
import numpy as np
import vstool
import MolIO
import pandas as pd
from rdkit import Chem

logger = vstool.setup_logger()

parser = argparse.ArgumentParser(prog='molecule-dynamics', description=__doc__.strip())
parser.add_argument('sdf', help="Path to a SDF file contains best docking pose for each cluster", type=vstool.check_file)
parser.add_argument('pdb', help="Path to a PDF file contains the structure for the docking target", type=vstool.check_file)
parser.add_argument('-o', '--outdir', default='.', type=vstool.mkdir,
                    help="Path to a directory for saving output files, default: %(default)s")
parser.add_argument('-t', '--time', type=float, default=50, help="MD simulation time, default: %(default)s ns.")
parser.add_argument('-c', "--cpu", type=int, default=4, help="Number of CPUs can be used, default: %(default)s.")
parser.add_argument('-g', "--gpu", type=int, default=4, help="Number of GPUs can be used, default: %(default)s.")
parser.add_argument('-q', '--quiet', help='Process data quietly without debug and info message', action='store_true')
parser.add_argument('-v', '--verbose', help='Process data verbosely with debug and info message',
                    action='store_true')

parser.add_argument('--openmm_simulate', help='Path to openmm_simulate executable')

parser.add_argument('--task', type=int, default=0, help="ID associated with this task")
parser.add_argument('--dependency', help="Dependency of this task")
parser.add_argument('--submit', action='store_true', help="Submit job to job queue instead of directly running it")
parser.add_argument('--hold', action='store_true',
                    help="Hold the submission without actually submit the job to the queue")
parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
vstool.setup_logger(quiet=args.quiet, verbose=args.verbose)

gpus = vstool.get_available_gpus(args.gpu, task=args.task)
setattr(args, 'outdir', args.sdf.parent / 'md')
setattr(args, 'result', args.sdf.parent / 'md.rmsd.csv')
setattr(args, 'cpu', 4 if args.debug else vstool.get_available_cpus(args.cpu))
setattr(args, 'gpu', gpus[:2] if args.debug else gpus)
os.chdir(args.outdir)


def pose_list():
    basename = args.outdir / args.sdf.with_suffix('').name
    batches = MolIO.batch_sdf(args.sdf, len(args.gpu), f'{basename}.')
    lists = [f'{basename}.{i}.txt' for i in args.gpu]

    if all([Path(x).exists() for x in lists]):
        logger.debug('Pose lists already exist, skip re-generating lists')
    else:
        for batch, li in zip(batches, lists):
            with open(li, 'w') as o:
                for i, s in enumerate(MolIO.parse_sdf(batch)):
                    if args.debug and i == len(args.gpu):
                        logger.debug(f'Debug mode was enabled, only first {len(args.gpu)} poses were saved to {li}')
                        break
                    out = s.sdf(args.outdir / f'{s.title}.sdf')
                    o.write(f'{out}\n')
    return lists


def md(txt):
    gpu = str(txt).rsplit('.')[-2]
    with open(txt) as f:
        for line in f:
            ligand = Path(line.strip())
            output = ligand.with_suffix(".trajectory.rmsd.csv")
            if output.exists():
                logger.debug(f'MD result for {ligand} already exists, skip re-doing MD')
            else:
                cmd = (f'{args.openmm_simulate} {ligand} {args.pdb} {ligand.with_suffix("")} '
                       f'--time {0.01 if args.debug else args.time} --devnum {gpu} &> {ligand.with_suffix(".log")}')
                p = cmder.run(cmd, exit_on_error=False)
                if p.returncode:
                    logger.error(f'Failed to run md on {ligand}')    


def submit():
    data = {'bin': Path(vstool.check_exe('python')).parent, 'outdir': args.outdir}
    env = ['source {bin}/activate', '', 'cd {outdir} || {{ echo "Failed to cd into {outdir}!"; exit 1; }}', '', '']

    cmd = ['molecule-dynamics', str(args.sdf), str(args.pdb),
           f'--outdir {args.outdir}', f'--time {args.time} ',
           f'--cpu {args.cpu}', f'--gpu {args.gpu}', f'--task {args.task}',
           f'--openmm_simulate {args.openmm_simulate}']

    vstool.submit('\n'.join(env).format(**data) + f' \\\n  '.join(cmd),
                  cpus_per_task=args.cpu, gpus_per_task=args.gpu, job_name=args.name,
                  day=args.day, hour=args.hour, array=f'1-{args.gpu}', partition=args.partition,
                  email=args.email, mail_type=args.email_type, log='%x.%j.%A.%a.log',
                  script='docking.sh', hold=args.hold)


def parse_rmsd():
    df, suffix = [], '.trajectory.rmsd.csv'
    for x in args.outdir.glob(f'*{suffix}'):
        name = x.name.removesuffix(suffix)
        try:
            s = next(MolIO.parse_sdf(f'{name}.sdf'))
            score = s.score
        except Exception as e:
            logger.error(f'Failed to retrieve docking score from {name}.sdf due to {e}')
            score = np.nan
        with open(x) as f:
            pose, rmsd = f.read().strip().split(',')
            try:
                rmsd = float(rmsd)
            except Exception as e:
                logger.error(f'Failed to retrieve RMSD from {x} due to {e}')
                rmsd = np.nan
            df.append({'pose': name, 'score': score, 'RMSD': rmsd})
    return df


@vstool.profile(task=args.task, status=125)
def main():
    if args.result.exists():
        vstool.debug_and_exit(f'MD skipped since results already exist\n', task=args.task, status=125)
    else:
        if args.submit or args.hold or args.dependency:
            submit()
        else:
            lists = pose_list()
            vstool.parallel_cpu_task(md, lists, processes=len(lists))

            df = parse_rmsd()
            if df:
                df = pd.DataFrame(df)
                df.to_csv(args.result, index=False, float_format='%.6f')

            cmder.run('rm -f *.txt *.sdf *.csv *.log')


if __name__ == '__main__':
    main()
