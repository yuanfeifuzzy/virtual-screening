# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run molecule dynamics in an easy way using Desmond
"""

import os
import sys
import tempfile
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


SCRIPT = Path(__file__).parent / 'desmond_md.sh'

parser = argparse.ArgumentParser(prog='molecule-dynamics', description=__doc__.strip())
parser.add_argument('sdf', help="Path to a single SDF file", type=vstool.check_file)
parser.add_argument('receptor', help="Path to a PDB or MAE file contains the structure for the docking target",
                    type=vstool.check_file)
parser.add_argument('--time', type=float, default=50, help="MD simulation time, default: %(default)s nanosecond.")
parser.add_argument('--summary', help="Basename of a CSV file for saving MD summary")
parser.add_argument('--outdir', help="Path to a directory for saving output files",
                    default='.', type=vstool.mkdir)
parser.add_argument('--desmond', help="Path to Desmon installation directory",
                    default='/software/Desmond/2023.2', type=vstool.check_dir)

parser.add_argument('--task', type=int, default=0, help="ID associated with this task")
parser.add_argument('--name', help="Name of the docking job, default: %(default)s", default='MD')
parser.add_argument('--email', help='Email address for send status change emails')
parser.add_argument('--email-type', help='Email type for send status change emails, default: %(default)s',
                    default='ALL', choices=('BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'))
parser.add_argument('--day', type=int, default=0, help="Number of days the job needs to run, default: %(default)s")
parser.add_argument('--hold', action='store_true', help="Hold the submission without actually submit it to the queue")

parser.add_argument('--quiet', help='Enable quiet mode.', action='store_true')
parser.add_argument('--verbose', help='Enable verbose mode.', action='store_true')
parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)
setattr(args, 'time', 1 if args.debug else args.time)

md = Path(vstool.check_exe("python")).parent / 'molecule-dynamics'
desmond = args.desmond or '/software/Desmond/2023.2'
desmond_md = Path(__file__).parent / 'desmond_md.sh'
struct_convert = f'{desmond}/utilities/structconvert'


def submit():
    directives = ['#SBATCH --nodelist=gpu', '#SBATCH --gpus-per-task=4']
    source = f'source {Path(vstool.check_exe("python")).parent / "activate"}'
    cd = f'cd {args.outdir} || {{ echo "Failed to cd into {args.outdir}!"; exit 1; }}'
    
    cmds = [source, cd]
    cmd = ['vs-simulate', str(args.sdf), str(args.receptor),
           f'--outdir {args.outdir}', f'--time {args.time} ', f'--task {args.task}']
    if args.summary:
        cmd.append(f'--summary {args.summary}')
    cmds.append(vstool.qvd(cmd, args))
    
    vstool.submit('\n'.join(cmds), job_name=args.name, day=args.day or 7, hour=0, directives=directives,
                  email=args.email, mail_type=args.email_type, log='%x.log', script='md.sh', hold=args.hold)


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
        cmder.run(f'zip -r -j {archive} rmsd.csv md.eaf md-out.cms md_trj/', cwd=str(wd), fmt_cmd=False)


def simulate(sdf, mae=None):
    gpu = int(sdf.name.split('.')[-2]) - 1
    sdf = Path(sdf)

    n, wd = 0, vstool.mkdir(sdf.with_suffix(''))
    for m in MolIO.parse_sdf(sdf):
        if m.mol:
            archive = args.outdir / f'{m.title}.md.zip'
            if archive.exists():
                logger.debug(f'MD results for {m.title} already exists, skip re-simulating')
            else:
                logger.debug(f'Running MD for ligand {m.title}')
                cwd = vstool.mkdir(wd / m.title)
                sd = m.sdf(output=str(cwd / f'{m.title}.sdf'))
                pose, view = cwd / 'pose.mae', cwd / 'view.mae'
                cmder.run(f'{struct_convert} {sd} {pose}', exit_on_error=False)
                cmder.run(f'cat {mae} {pose} > {view}', exit_on_error=False)
                cuda = f'export CUDA_VISIBLE_DEVICES={gpu}'
                p = cmder.run(f'{cuda} && {desmond_md} {cwd} {view} {int(1000 * args.time)}', exit_on_error=False)
                # env = os.environ.copy()
                # env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                # p = cmder.run(f'{desmond_md} {cwd} {view} {int(1000 * args.time)}', exit_on_error=False, env=env)
                if p.returncode:
                    vstool.error_and_exit(f'Failed to run MD using {sd}', task=args.task)
                parse(cwd)
                if not args.debug:
                    cmder.run(f'rm -r {cwd}')

            n += 1
            if args.debug and n == 2:
                break


def post_md():
    df = []
    for x in args.outdir.glob('*.rmsd.csv'):
        df.append(pd.read_csv(x))
        os.unlink(x)
    if df:
        df = pd.concat(df)
        columns = ['ligand', 'score']
        df = df[columns + [c for c in df.columns if c not in columns]]
    return df


def main():
    if args.day or args.hold:
        submit()
    else:
        start = time.time()
        vstool.task_update(args.task, 130)
        wd = vstool.mkdir(tempfile.mkdtemp(prefix='md.', suffix=args.sdf.with_suffix('').name))
        mae = args.outdir / args.receptor.with_suffix('.mae').name

        if not mae.exists():
            cmder.run(f'{struct_convert} {args.receptor} {mae}', exit_on_error=True)

        sdfs = MolIO.batch_sdf(str(args.sdf), 4, wd / 'md.')
        vstool.parallel_cpu_task(simulate, sdfs, processes=min(4, len(sdfs)), mae=mae)

        if args.summary:
            df = post_md()
            print(df)
            df.to_csv(args.outdir / args.summary, index=False, float_format='%.4f')
            logger.debug(f'MD summary was successfully saved to {args.outdir / args.summary}')

        if not args.debug:
            logger.debug(f'Cleaning up directory {args.scratch} that contains intermediate files')
            cmder.run(f'rm -r {wd}')

        t = str(timedelta(seconds=time.time() - start))
        vstool.debug_and_exit(f'MD complete in {t.split(".")[0]}\n', task=args.task, status=135)
        vstool.debug_and_exit(f'Virtual screening complete\n', task=args.task, status=140)


if __name__ == '__main__':
    main()

