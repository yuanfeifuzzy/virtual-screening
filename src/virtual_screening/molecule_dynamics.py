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
parser.add_argument('pdb', help="Path to a PDF file contains the structure for the docking target",
                    type=vstool.check_file)
parser.add_argument('-t', '--time', type=float, default=50, help="MD simulation time, default: %(default)s ns.")
parser.add_argument('-e', '--exe', help='Path to MD executable, default: %(default)s',
                    type=vstool.check_exe, default=SCRIPT)
parser.add_argument('--scratch', help="Path to the scratch directory, default: %(default)s",
                    default=Path(os.environ.get('SCRATCH', '/scratch')))
parser.add_argument('--summary', help='Path to a CSV file for saving MD summary results, default: %(default)s',
                    default='md.summary')

parser.add_argument('--nodes', type=int, default=0, help="Number of nodes, default: %(default)s.")
parser.add_argument('--project', help='The nmme of project you would like to be charged')
parser.add_argument('--email', help='Email address for send status change emails')
parser.add_argument('--email-type', help='Email type for send status change emails, default: %(default)s',
                    default='ALL', choices=('NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'))
parser.add_argument('--delay', help='Hours need to delay running the job.', type=int, default=0)

parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)

setattr(args, 'scratch', vstool.mkdir(args.scratch / f'{args.sdf.parent.name}'))


def cmding(env, cmd, args):
    if args.debug:
        cmd.append('--debug')
    return env + ' \\\n  '.join(cmd)


def submit():
    hostname, sdf, outdir = socket.gethostname(), args.sdf, args.scratch
    ntasks_per_node = 4 if 'frontera' in hostname else 3
    ntasks = args.nodes * ntasks_per_node
    gpu_queue = 'rtx' if 'frontera' in hostname else 'gpu-a100'

    lcmd = ['module load launcher_gpu', 'export LAUNCHER_WORKDIR={outdir}',
            'export LAUNCHER_JOB_FILE={outdir}/docking.commands.txt', '', '${{LAUNCHER_DIR}}/paramrun', '']
    cmds, output = [], outdir / 'md.commands.txt'
    program = Path(vstool.check_exe("python")).parent / 'molecule-dynamics'

    for s in MolIO.parse_sdf(sdf):
        if s.mol:
            d = outdir / s.title
            d.mkdir(exist_ok=True)
            out = s.sdf(output=str(d / f'{s.title}.sdf'))
            cmd = f'{program} {out} {args.pdb} --time {args.time} --exe {args.exe}'
            if args.debug:
                cmd = f'{cmd} --debug'
            cmds.append(cmd)

    if cmds:
        with output.open('w') as o:
            o.writelines(f'{cmd}\n' for cmd in cmds)
        logger.debug(f'Successfully saved {len(cmds)} launch commands to {output}\n')

        code, job_id = vstool.submit('\n'.join(lcmd).format(outdir=str(outdir)), nodes=args.nodes, ntasks=ntasks,
                                     ntasks_per_node=ntasks_per_node, job_name='md', day=0 if args.debug else 1,
                                     hour=8 if args.debug else 20, partition=gpu_queue, email=args.email,
                                     mail_type=args.email_type, log=outdir / 'md.log', mode='append',
                                     script=outdir / 'md.sh', delay=args.delay)

        env = (f'source {Path(vstool.check_exe("python")).parent}/activate\n\n'
               f'cd {outdir} || {{ echo "Failed to cd into {outdir}!"; exit 1; }}\n\n')
        cmd = f'post-md {outdir} --summary {args.summary}'
        if args.debug:
            cmd = f'{cmd} --debug'
        vstool.submit(env+cmd,
                      nodes=1, job_name='post-md', hour=1, minute=30,
                      partition='flex' if 'frontera' in hostname else 'vm-small',
                      email=args.email, mail_type=args.email_type,
                      log=outdir / 'md.log', mode='append', script=outdir / 'post.md.sh',
                      dependency=f'afterok:{job_id}', delay=args.delay, project=args.project)


def parse(wd):
    eaf, sdf, output = 'md.eaf', f'{wd}.sdf', f'{wd}.rmsd.csv'
    logger.debug(f'Parsing {wd}/{eaf} ...')
    rmsd, flag, n = [], 0, 1
    with eaf.open() as f:
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
        df = {'ligand': wd.name, 'score': score, 'rmsd': rmsd, 'rmsd_min': np.min(rmsd), 'rmsd_max': np.max(rmsd)}
        df = pd.DataFrame([df])
        df.to_csv('rmsd.csv', index=False, float_format='%.4f')
        logger.debug(f'Successfully saved rmsd results to {wd}/rmsd.csv')


def main():
    if args.nodes:
        submit()
    else:
        n = MolIO.count_sdf(args.sdf)
        if n != 1:
            vstool.error_and_exit(f'SDF file {args.sdf} contains more than 1 records (n={n:,}), cannot continue')

        wd = args.sdf.parent
        os.chdir(wd)

        receptor = args.pdb.with_suffix('.mae').name
        pose = args.sdf.with_suffix('.mae').name
        view = args.sdf.with_suffix('.pose.view.mae').name

        p = cmder.run(f'{schrodinger}/utilities/structconvert {pdb} {receptor}', exit_on_error=False, cwd=str(wd))
        if p.returncode:
            raise RuntimeError(f'Failed to convert {args.pdb.name} to {receptor}')

        p = cmder.run(f'{schrodinger}/utilities/structconvert {sdf} {pose}', exit_on_error=False, cwd=str(wd))
        if p.returncode:
            raise RuntimeError(f'Failed to convert {args.sdf.name} to {pose}')

        p = cmder.run(f'cat {receptor} {pose} > {view}', exit_on_error=False, cwd=str(wd))
        if p.returncode:
            raise RuntimeError(f'Failed to concatenate {receptor.name} and {pose}')

        p = cmder.run(f'{args.exe} {wd} {view} {args.time}', exit_on_error=False, debug=True, cwd=str(wd))
        if p.returncode:
            vstool.error_and_eixt(f'Failed to run MD for {args.sdf}')
        else:
            parse(wd)


if __name__ == '__main__':
    main()
