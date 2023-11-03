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
                    type=vstool.check_exe, default=MD)
parser.add_argument('--scratch', help="Path to the scratch directory, default: %(default)s",
                    default=Path(os.environ.get('SCRATCH', '/scratch')))

parser.add_argument('--nodes', type=int, default=0, help="Number of nodes, default: %(default)s.")
parser.add_argument('--email', help='Email address for send status change emails')
parser.add_argument('--email-type', help='Email type for send status change emails, default: %(default)s',
                    default='ALL', choices=('NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'))
parser.add_argument('--dependency', help='Dependency of current job if submit to the queue.')
parser.add_argument('--log', help='Log file for current job if submit to the queue.')
parser.add_argument('--summary', help='Path to a CSV file for saving MD summary results.')
parser.add_argument('--delay', help='Hours need to delay running the job.', type=int, default=0)

parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)

setattr(args, 'scratch', vstool.mkdir(args.scratch / f'{args.sdf.parent.name}'))


def submit():
    hostname, sdf, outdir = socket.gethostname(), args.sdf, args.scratch
    ntasks_per_node = 4 if 'frontera' in hostname else 3
    ntasks = args.nodes * ntasks_per_node
    gpu_queue = 'rtx' if 'frontera' in hostname else 'gpu-a100'

    lcmd = ['module load launcher_gpu', 'export LAUNCHER_WORKDIR={outdir}',
            'export LAUNCHER_JOB_FILE={outdir}/docking.commands.txt', '', '${{LAUNCHER_DIR}}/paramrun', '']

    logger.debug(f'Splitting {sdf} into {ntasks} batches ...')
    batches = MolIO.batch_sdf(sdf, ntasks, outdir / 'batch.')

    cmds, output = [], outdir / 'md.commands.txt'
    program = Path(vstool.check_exe("python")).parent / 'docking'

    for batch in batches:
        cmd = f'{program} {batch} {args.pdb} --time {args.time} --exe {args.exe} --scratch {args.scratch}'
        if args.debug:
            cmd = f'{cmd} --debug'
        cmds.append(cmd)

    if cmds:
        with output.open('w') as o:
            o.writelines(f'{cmd}\n' for cmd in cmds)
        logger.debug(f'Successfully saved {len(cmds)} launch commands to {output}')

        vstool.submit('\n'.join(lcmd).format(outdir=str(outdir)), nodes=args.nodes, ntasks=ntasks,
                      ntasks_per_node=ntasks_per_node, job_name='md', day=0 if args.debug else 1,
                      hour=8 if args.debug else 20, partition=gpu_queue, email=args.email, mail_type=args.email_type,
                      log=args.log or 'md.log', mode='append' if args.log else '',
                      script=outdir / 'md.sh', dependency=args.dependency, delay=args.delay)


def parse(wd):
    eaf, sdf, output = wd / 'md.eaf', f'{wd}.sdf', f'{wd}.rmsd.csv'
    logger.debug(f'Parsing {eaf} ...')
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
            logger.error(f'Failed to get docking score from {sdf}')
        df = {'ligand': wd.name, 'score': score, 'rmsd': rmsd,
              'rmsd_min': np.min(rmsd), 'rmsd_max': np.max(rmsd)}
        df = pd.DataFrame([df])
        df.to_csv(output, index=False, float_format='%.4f')
        df.to_csv('rmsd.csv', index=False, float_format='%.4f')
        logger.debug(f'Successfully saved rmsd results to ')

        cmder.run(f'zip -r {args.sdf.with_suffix(".md.zip")} rmsd.csv md.eaf md-out.cms md_trj/', cwd=str(wd))
        if not args.debug:
            cmder.run(f'rm -r {wd}')

    sdfs = wd.parent.glob('*.sdf')
    running, done = [], []
    for sdf in sdfs:
        out = sdf.with_suffix('.rmsd.csv')
        if out.exists():
            done.append(out)
        else:
            running.append(sdf)

    if done and not running:
        logger.debug('All MD jobs are done, summarizing MD results ...\n')
        summary = args.summary or wd.parent / 'md.summary.csv'
        df = [pd.read_csv(out) for out in done]
        df = pd.concat(df)
        df.to_csv(summary, index=False, float_format='%.4f')
        logger.debug(f'All done, MD summary was saved to {summary}\n')
    else:
        if running:
            logger.debug(f'The following {len(running)} SDF(s) are still processing or pending for processing:')
            for run in running:
                logger.debug(f'  {run}')


def main():
    if args.nodes:
        submit()
    else:
        outdir = args.sdf.parent
        wd = vstool.mkdir(outdir / args.sdf.with_suffix(''))
        os.chdir(wd)

        receptor = args.pdb.with_suffix('.mae').name
        pose = args.sdf.with_suffix('.mae').name
        view = args.sdf.with_suffix('.pose.view.mae').name

        try:
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
            if p.returncode == 0:
                parse(wd)
        finally:
            if not debug:
                cmder.run(f'rm -f {receptor} {pose} {view}', log_cmd=False)


if __name__ == '__main__':
    main()
