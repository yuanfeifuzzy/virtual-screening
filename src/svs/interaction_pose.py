#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filter ligand target interaction using Schrodinger pose filter script.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import timedelta

import pandas as pd
import cmder
import subprocess
import utility
from rdkit import Chem
from pandarallel import pandarallel

logger = utility.setup_logger()


def pdb_to_sdf(pdb, sdf=''):
    sdf = sdf if sdf else str(Path(pdb).with_suffix('.sdf'))
    cmder.run(f'obabel {sdf} -O {sdf}')
    return sdf


def filter_interaction(sdf, pdb, residue_number, output='interaction.pose.sdf', 
                       schrodinger='', quiet=False, verbose=False):
    utility.setup_logger(quiet=quiet, verbose=verbose)

    schrodinger = schrodinger or os.environ.get('SCHRODINGER', '')
    if not schrodinger:
        logger.error('No schrodinger was provided and cannot find SCHRODINGER')
        sys.exit(1)
        
    receptor = pdb_to_sdf(pdb, str(Path(pdb).with_suffix('.sdf').name))
    view = pose.view.sdf
    cmder.run(f'cat {receptor} {sdf} > {view}')
    options = ' '.join([f'-asl {n} -hbond {i}' for i, n in enumerate(residue_number, 1)])
    
    log = Path(output).with_suffix('.log')
    cmder.run(f'{schrodinger}/run pose_filter.py {options} {view} {output} -LOCAL -WAIT > {log}')


def main():
    parser = argparse.ArgumentParser(prog='pose-interaction', description=__doc__.strip())
    parser.add_argument('sdf', help="Path to a SDF file stores docking poses")
    parser.add_argument('pdb', help="Path to a PDB file stores receptor structure")
    parser.add_argument('residue', help="residue numbers that interact with ligand via hydrogen bond", nargs='+')
    parser.add_argument('-o', '--output', help="Path to a SDF file for saving output poses")
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
    if args.submit or args.hold:
        prog, wd = parser.prog, utility.make_directory(args.wd, task=args.task, status=-1)
        activation = utility.check_executable(prog).strip().replace(prog, 'activation', task=args.task, status=-2)
        venv = f'source {activation}\ncd {wd}'
        cmdline = utility.format_cmd(prog, args, ['sdf', 'pdb', 'residue'],
                                     ['output', 'schrodinger', 'quiet', 'verbose', 'task'])
        # options = utility.cmd_options(vars(args), excludes=['submit', 'hold'])
        # cmdline = fr'sd \\\n  {args.source}\\\n  --outdir {outdir}\\\n  {options}'

        return_code, job_id = utility.submit(cmdline, venv=venv, name=prog, day=0, hour=16, hold=args.hold,
                                             script=f'{prog}.sh')
        utility.update_status(return_code, job_id, args.task, 95)
        sys.exit(0)
    else:
        try:
            start = time.time()
            output = args.output or Path(sdf).resolve().parent / 'interaction.pose.sdf'
            filter_interaction(args.sdf, args.pdb, args.residue, output=output, schrodinger=args.schrodinger,
                               quiet=args.quiet, verbose=args.verbose)
            t = str(timedelta(seconds=time.time() - start))
            utility.debug_and_exit(f'Filter interaction pose complete in {t.split(".")[0]}\n',
                                   task=args.task, status=110)
        except Exception as e:
            utility.error_and_exit(f'Filter interaction pose failed due to {e}\n', task=args.task, status=-110)


if __name__ == '__main__':
    main()
