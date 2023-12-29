# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run molecule dynamics in an easy way
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

parser = argparse.ArgumentParser(prog='vs-md', description=__doc__.strip())
parser.add_argument('sdf', help="Path to a single SDF file", type=vstool.check_file)
parser.add_argument('receptor', help="Path to a PDB or mae file contains the structure for the docking target",
                    type=vstool.check_file)
parser.add_argument('--time', type=float, default=200, help="MD simulation time, default: %(default)s ns.")
parser.add_argument('--outdir', help="Path to a directory for saving output files",
                    default='.', type=vstool.mkdir)
parser.add_argument('--desmond', help="Path to Desmon installation directory",
                    default='/work/02940/ztan818/ls6/software/DESRES/2023.2', type=vstool.check_dir)

parser.add_argument('--quiet', help='Enable quiet mode.', action='store_true')
parser.add_argument('--verbose', help='Enable verbose mode.', action='store_true')
parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)
setattr(args, 'time', 1 if args.debug else args.time)
setattr(args, 'scratch', vstool.mkdir(f"{os.environ.get('SCRATCH', '/scratch')}/{args.outdir.name}"))

desmond_md = Path(__file__).parent / 'desmond_md.sh'
struct_convert = f'{args.desmond}/utilities/structconvert'


def parse(wd, archive):
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
        summary = f'{args.outdir}/{wd.name}.md.summary.csv'
        df.to_csv(summary, index=False, float_format='%.4f')
        logger.debug(f'Successfully saved rmsd results to {summary}.rmsd.csv')

        cmder.run(f'zip -r {archive}.md.zip rmsd.csv md.eaf md-out.cms md_trj/', cwd=str(wd))


def simulate(sdf, mae, archive):
    pose, view = sdf.with_suffix('.mae'), sdf.with_suffix('.view.mae')
    cmder.run(f'{struct_convert} {sdf} {pose}', exit_on_error=False)
    cmder.run(f'cat {mae} {pose} > {view}', exit_on_error=False)
    p = cmder.run(f'{desmond_md} {sdf.parent} {view} {int(1000*args.time)}', exit_on_error=False)
    if p.returncode:
        vstool.error_and_exit(f'Failed to run MD for {sdf}')
    parse(sdf.parent, archive)


def main():
    wd = vstool.mkdir(tempfile.mkdtemp(prefix='md.', suffix=f'.{args.sdf.withsuffix("").name}'))
    mae = args.scratch / args.receptor.with_suffix('.mae').name

    if not mae.exists():
        cmder.run(f'{struct_convert} {args.receptor} {mae}', exit_on_error=True)

    n = 0
    for m in MolIO.parse_sdf(args.sdf):
        if m.mol:
            archive = args.scratch / f'{m.title}.md.zip'
            if archive.exists():
                logger.debug(f'MD results for {m.title} already exists, skip re-simulating')
            else:
                cwd = vstool.mkdir(wd / m.title)
                sdf = m.sdf(output=str(cwd / f'{m.title}.sdf'))
                simulate(sdf, mae, archive)
            n += 1
            if args.debug and n == 3:
                break


if __name__ == '__main__':
    main()
