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

logger = vstool.setup_logger(verbose=True)
VENV = '/work/08944/fuzzy/share/software/virtual-screening/venv'
MD = f'{VENV}/lib/python3.11/site-packages/virtual_screening/desmond_md.sh'

parser = argparse.ArgumentParser(prog='molecule-dynamics', description=__doc__.strip())
parser.add_argument('sdf', help="Path to a single SDF file", type=vstool.check_file)
parser.add_argument('pdb', help="Path to a PDF file contains the structure for the docking target", type=vstool.check_file)
parser.add_argument('-t', '--time', type=float, default=50, help="MD simulation time, default: %(default)s ns.")
parser.add_argument('-e', '--exe', help='Path to Schrodinger Suite root directory, defualt: %(default)s',
                    type=vstool.check_exe, default=MD)

parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
vstool.setup_logger(verbose=True)


def main():
    wd = args.sdf.parent
    outdir = vstool.mkdir(wd / args.sdf.with_suffix(''))
    receptor = wd / args.pdb.with_suffix('.mae')
    pose = args.sdf.with_suffix('.mae')
    view = args.sdf.with_suffix('.pose.view.mae')
    
    try:
        p = cmder.run(f'{schrodinger}/utilities/structconvert {pdb} {receptor}', exit_on_error=False)
        if p.returncode:
            raise RuntimeError(f'Failed to convert {args.pdb.name} to {receptor.name}')
        
        p = cmder.run(f'{schrodinger}/utilities/structconvert {sdf} {pose}', exit_on_error=False)
        if p.returncode:
            raise RuntimeError(f'Failed to convert {args.sdf.name} to {pose.name}')
        
        p = cmder.run(f'cat {receptor} {pose} > {view}', exit_on_error=False)
        if p.returncode:
            raise RuntimeError(f'Failed to concatenate {receptor.name} and {pose.name}')

        p = cmder.run(f'{args.exe} {outdir} {view} {args.time}', exit_on_error=False, debug=True)
        if p.returncode == 0:
            with open(output, 'w') as o:
                f = MolIO.parse_sdf(out)
                next(f)
                for s in f:
                    o.write(s.sdf(title=f'{s.title}_{s.score}'))
    finally:
        if not debug:
            cmder.run(f'rm -f {receptor} {pose} {view}', log_cmd=False)


if __name__ == '__main__':
    main()
