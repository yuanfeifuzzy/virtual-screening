#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Get top poses from docking results
"""

import sys
import time
import argparse
import subprocess
import traceback
from pathlib import Path
from datetime import timedelta

import pandas as pd

import utility
from rdkit import Chem
from pandarallel import pandarallel
from svs import sdf_io

logger = utility.setup_logger()


def get_pose_from_dlg(dlg, idx):
    with subprocess.Popen([f'mk_export.py', dlg, '-'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        with Chem.ForwardSDMolSupplier(p.stdout) as f:
            for i, mol in enumerate(f):
                if i == idx:
                    return mol


def get_pose_from_sdf(path, idx):
    with Chem.ForwardSDMolSupplier(path) as f:
        for i, mol in enumerate(f):
            if i == idx:
                return mol


def get_pose(path, idx):
    if path.endswith('.dlg'):
        pose = get_pose_from_dlg(path, idx)
    else:
        pose = get_pose_from_sdf(path, idx)
    return pose


def top_pose(score, top_percent=10, output='top.pose.sdf'):
    logger.debug('Loading docking results')
    df = utility.read(score)
    logger.debug(f'Successfully loaded docking results for {df.shape[0]:,} ligands')

    logger.debug('Sorting docking scores')
    df.sort_values('score', inplace=True)
    n = int(df.shape[0] * top_percent / 100)
    logger.debug(f'Successfully get top {top_percent}% (n={df.shape[0]:,}) poses')

    logger.debug(f'Retrieving best top poses ...')
    sdf = str(score).replace('.scores.parquet', '.poses.sdf.gz')
    with open(output, 'w') as o:
        for i, pose in enumerate(sdf_io.parse(sdf)):
            o.write(f'{pose.sdf().rstrip()}')
            if i == n:
                break
    logger.debug(f'Successfully saved {n:,} best top poses to {output}')


def main():
    parser = argparse.ArgumentParser(prog='top-pose', description=__doc__.strip())
    parser.add_argument('path', help="Path to a parquet file contains docking scores")
    parser.add_argument('-t', '--top', help="Percentage of top poses need to be retained", default=10, type=float)
    parser.add_argument('-o', '--output', help="Path to a output for saving top poses", default='top.pose.sdf')

    parser.add_argument('-q', '--quiet', help='Process data quietly without debug and info message',
                        action='store_true')
    parser.add_argument('-v', '--verbose', help='Process data verbosely with debug and info message',
                        action='store_true')

    parser.add_argument('--wd', help="Path to work directory", default='.')
    parser.add_argument('--task', type=int, default=0, help="ID associated with this task")
    
    parser.add_argument('--submit', action='store_true', help="Submit job to job queue instead of directly running it")
    parser.add_argument('--hold', action='store_true',
                        help="Hold the submission without actually submit the job to the queue")

    args = parser.parse_args()
    utility.setup_logger(quiet=args.quiet, verbose=args.verbose)
    
    utility.submit_or_skip(parser.prog, args, ['path'], ['top', 'output', 'quiet', 'verbose', 'wd', 'task'], day=1)

    try:
        start = time.time()
        output = Path(args.output) or Path(sdf).resolve().parent / 'top.pose.sdf'
        if output.exists():
            utility.debug_and_exit(f'Top pose already exists, skip re-processing\n', task=args.task, status=95)

        top_pose(args.path, top_percent=args.top, output=str(output))
        
        t = str(timedelta(seconds=time.time() - start))
        utility.debug_and_exit(f'Get top pose complete in {t.split(".")[0]}\n', task=args.task, status=95)
    except Exception as e:
        utility.error_and_exit(f'Get top pose failed due to\n{e}\n\n{traceback.format_exc()}\n',
                               task=args.task, status=-95)


if __name__ == '__main__':
    main()
