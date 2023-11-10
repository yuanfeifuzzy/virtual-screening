# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run molecule dynamics in an easy way
"""

import os
import sys
import time
import socket
import argparse
import traceback
import importlib
from pathlib import Path
from datetime import timedelta
from multiprocessing import Pool, Queue

import cmder
import numpy as np
import vstool
import MolIO
import pandas as pd
from rdkit import Chem


SCRIPT = importlib.resources.files(__package__) / 'desmond_md.sh'

parser = argparse.ArgumentParser(prog='molecule-dynamics', description=__doc__.strip())
parser.add_argument('sdf', help="Path to a single SDF file", type=vstool.check_file)
parser.add_argument('receptor', help="Path to a PDB or mae file contains the structure for the docking target",
                    type=vstool.check_file)
parser.add_argument('--time', type=float, default=50, help="MD simulation time, default: %(default)s ns.")
parser.add_argument('--scratch', help="Path to the scratch directory, default: %(default)s",
                    default=os.environ.get('SCRATCH', '/scratch'), type=vstool.mkdir)
parser.add_argument('--outdir', help="Path to a directory for saving output files",
                    default='.', type=vstool.mkdir)
parser.add_argument('--summary', help="Basename of a CSV file for saving MD summary, default: %(default)s",
                    default='md.summary.csv')

parser.add_argument('--nodes', type=int, default=0, help="Number of nodes, default: %(default)s.")
parser.add_argument('--project', help='The nmme of project you would like to be charged')
parser.add_argument('--email', help='Email address for send status change emails')
parser.add_argument('--email-type', help='Email type for send status change emails, default: %(default)s',
                    default='ALL', choices=('NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'))
parser.add_argument('--delay', help='Hours need to delay running the job.', type=int, default=0)
parser.add_argument('--hold', help='Only generate submit script but hold for submitting', action='store_true')

parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)
setattr(args, 'scratch', vstool.mkdir(args.scratch / f'{args.outdir.name}'))

md = Path(vstool.check_exe("python")).parent / 'molecule-dynamics'
desmond_md = Path(__file__).parent / 'desmond_md.sh'
structconvert = '/work/02940/ztan818/ls6/software/DESRES/2023.2/utilities/structconvert'


def submit():
    summary = args.outdir / args.summary
    if summary.exists():
        vstool.debug_and_exit('MD summary results already exists, skip re-simulating')

    hostname = socket.gethostname()
    ntasks_per_node = 4 if 'frontera' in hostname else 3
    ntasks = args.nodes * ntasks_per_node
    queue = 'rtx' if 'frontera' in hostname else 'gpu-a100'
    queue = f'{queue}-dev' if args.debug else queue
    devices = [str(x) for x in range(ntasks_per_node)] * args.nodes
    
    basename = args.sdf.with_suffix('').name
    sdfs = MolIO.batch_sdf(str(args.sdf), ntasks, str(args.scratch / f'{basename}.'))
    commands = args.scratch / 'md.commands.txt'
    with open(commands, 'w') as o:
        for sdf, device in zip(sdfs, devices):
            cuda = f'export CUDA_VISIBLE_DEVICES={device}'
            cmd = (f'{cuda} && {md} {sdf} {args.receptor} --time {args.time} --outdir {args.outdir} '
                   f'--scratch {args.scratch}')
            if args.debug:
                cmd = f'{cmd} --debug'
            o.write(f'{cmd}\n')
    logger.debug(f'Successfully saved {len(sdfs)} md launch commands to {commands}')

    lcmd = ['', 'module load launcher_gpu', f'export LAUNCHER_WORKDIR={args.scratch}',
            f'export LAUNCHER_JOB_FILE={args.scratch}/md.commands.txt', '"${LAUNCHER_DIR}"/paramrun', '']
    lcmd = '\n'.join(lcmd)

    source = f'source {Path(vstool.check_exe("python")).parent / "activate"}'
    cd = f'cd {args.outdir} || {{ echo "Failed to cd into {args.outdir}!"; exit 1; }}'
    
    cmd = f'post-md {args.outdir} {args.summary} --scratch {args.scratch}'
    if args.debug:
        cmd = f'{cmd} --debug'
    vstool.submit('\n'.join([source, cd, lcmd, cmd]),
                  nodes=args.nodes, job_name='md', ntasks=ntasks, ntasks_per_node=ntasks_per_node,
                  day=0 if args.debug else 1, hour=1 if args.debug else 23, minute=59,
                  partition=queue, email=args.email, mail_type=args.email_type,
                  log='md.log', script=args.outdir / 'md.sh', delay=args.delay,
                  project=args.project, hold=args.hold)


def parse(wd):
    eaf, sdf = wd / 'md.eaf', wd / f'{wd.name}.sdf'
    logger.debug(f'Parsing {eaf} ...')
    rmsd, flag, n = [], 0, 1
    with open(eaf) as f:
        for line in f:
            if "RMSD" in line:
                flag = 1
                n = 1
            elif flag == 1:
                if n == 0:
                    if 'FitBy = "(protein)"' in line:
                        flag = 2
                        n = 2
                    else:
                        flag = 0
                        n = 1
                else:
                    n -= 1
            elif flag == 2:
                if n == 0:
                    rmsd = line.strip().split("= [")[1].replace(" ]", "").split(" ")
                    break
                else:
                    n -= 1

    if rmsd:
        rmsd = np.array(rmsd, dtype=float)
        try:
            s = next(MolIO.parse_sdf(sdf))
            score = s.score if s else np.nan
        except Exception as e:
            score = np.nan
            logger.error(f'Failed to get docking score from {wd}/{sdf} due to {e}')

        df = pd.Series(rmsd).describe().to_dict()
        df = {f'RMSD_{k}': v for k, v in df.items()}
        df['ligand'], df['score'] = wd.name, score
        df = pd.DataFrame([df])
        columns = ['ligand', 'score']
        df = df[columns + [c for c in df.columns if c not in columns]]
        df.to_csv(wd / 'rmsd.csv', index=False, float_format='%.4f')
        df.to_csv(args.outdir / f'{wd.name}.rmsd.csv', index=False, float_format='%.4f')
        logger.debug(f'Successfully saved rmsd results to {args.outdir / wd.name}.rmsd.csv')

        archive = args.outdir / f'{wd.name}.md.zip'
        cmder.run(f'zip -r {archive} rmsd.csv md.eaf md-out.cms md_trj/', cwd=str(wd))


def simulate(sdf, mae):
    sdf = Path(sdf)
    pose, view = sdf.with_suffix('.mae'), sdf.with_suffix('.view.mae')
    cmder.run(f'{structconvert} {sdf} {pose}', exit_on_error=False)
    cmder.run(f'cat {mae} {pose} > {view}', exit_on_error=False)
    p = cmder.run(f'{desmond_md} {sdf.parent} {view} {int(1000*args.time)}', exit_on_error=False)
    if p.returncode:
        vstool.error_and_exit(f'Failed to run MD for {sdf}')
    parse(sdf.parent)


def main():
    if args.nodes:
        submit()
    else:
        wd = vstool.mkdir(args.scratch / args.sdf.with_suffix(''))
        mae = args.scratch / args.receptor.with_suffix('.mae').name

        if not mae.exists():
            cmder.run(f'{structconvert} {args.receptor} {mae}', exit_on_error=True)

        n = 0
        for m in MolIO.parse_sdf(args.sdf):
            if m.mol:
                archive = args.outdir / f'{m.title}.md.zip'
                if archive.exists():
                    logger.debug(f'MD results for {m.title} already exists, skip re-simulating')
                else:
                    cwd = vstool.mkdir(wd / m.title)
                    sdf = m.sdf(output=str(cwd / f'{m.title}.sdf'))
                    simulate(sdf, mae)
                n += 1
                if args.debug and n == 3:
                    break


if __name__ == '__main__':
    main()
