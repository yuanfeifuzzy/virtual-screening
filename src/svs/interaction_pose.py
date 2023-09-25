#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filter ligand target interaction using Schrodinger pose filter script.
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import timedelta

import pandas as pd
import cmder
import subprocess
import utility
from rdkit import Chem
from pandarallel import pandarallel

from svs import tools

logger = utility.setup_logger()


def filter_interaction(sdf, pdb, residue_number, output='interaction.pose.sdf', 
                       schrodinger='', quiet=False, verbose=False, task=0):
    utility.setup_logger(quiet=quiet, verbose=verbose)

    schrodinger = schrodinger or os.environ.get('SCHRODINGER', '')
    if not schrodinger:
        utility.error_and_exit('No schrodinger was provided and cannot find SCHRODINGER', task=task, status=-2)
        
    receptor = str(Path(pdb).with_suffix('.sdf').name)
    cmder.run(f'{schrodinger}/utilities/structconvert {pdb} {receptor}', exit_on_error=True)
    # https://www.schrodinger.com/kb/286 $SCHRODINGER/run pv_convert.py -m combined.mae
    view = Path(sdf).resolve().parent / 'pose.view.sdf'
    # https://www.schrodinger.com/kb/1168#:~:text=Yes%2C%20you%20can%20create%20a,both%20compressed%20and%20uncompressed%20files.
    cmder.run(f'cat {receptor} {sdf} > {view}')
    options = ' '.join([f"-asl 'res.num {n}' -hbond {i}" for i, n in enumerate(residue_number, 1)])
    
    log = Path(output).with_suffix('.log')
    # RuntimeError: Could not extract atoms from Structure 36 (.mae) / Structure 72 (.sdf)
    cmder.run(f'{schrodinger}/run pose_filter.py {view} {output} {options} -WAIT -NOJOBID > {log}',
              fmt_cmd=False, exit_on_error=True)


def main():
    parser = argparse.ArgumentParser(prog='interaction-pose', description=__doc__.strip())
    parser.add_argument('sdf', help="Path to a SDF file stores docking poses")
    parser.add_argument('pdb', help="Path to a PDB file stores receptor structure")
    parser.add_argument('-r', '--residue', nargs='?', type=int,
                        help="Residue numbers that interact with ligand via hydrogen bond")
    parser.add_argument('-o', '--output', help="Path to a SDF file for saving output poses",
                        default='interaction.pose.sdf')
    parser.add_argument('-s', '--schrodinger', help='Path to Schrodinger Suite root directory')
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
    
    tools.submit_or_skip(parser.prog, args, ['sdf', 'pdb'],
                         ['residue', 'output', 'schrodinger', 'quiet', 'verbose', 'task'], day=0)
    
    try:
        start = time.time()
        output = Path(args.output) or Path(args.sdf).resolve().parent / 'interaction.pose.sdf'
        
        if output.exists():
            utility.debug_and_exit(f'Interaction pose already exists, skip re-processing\n', task=args.task, status=115)

        if args.residue:
            filter_interaction(args.sdf, args.pdb, args.residue, output=str(output), schrodinger=args.schrodinger,
                               quiet=args.quiet, verbose=args.verbose, task=args.task)
        else:
            cmder.run(f'cp {sdf} {output}')
        t = str(timedelta(seconds=time.time() - start))
        utility.debug_and_exit(f'Filter interaction pose complete in {t.split(".")[0]}\n',
                               task=args.task, status=115)
    except Exception as e:
        utility.error_and_exit(f'Filter interaction pose failed due to\n{e}\n\n{traceback.format_exc()}',
                               task=args.task, status=-115)


if __name__ == '__main__':
    main()
