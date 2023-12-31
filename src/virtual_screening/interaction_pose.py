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
import vstool
import MolIO

logger = vstool.setup_logger(verbose=True)


def interaction_pose(sdf, pdb, residue_number, output='interaction.pose.sdf', schrodinger='', task=0,
                     quiet=False, verbose=False):
    vstool.setup_logger(verbose=verbose, quiet=quiet)

    schrodinger = schrodinger or os.environ.get('SCHRODINGER', '')
    if not schrodinger:
        vstool.error_and_exit('No schrodinger was provided and cannot find SCHRODINGER', task=task, status=-2)

    receptor = str(Path(pdb).with_suffix('.mae').name)
    pose = str(Path(sdf).with_suffix('.mae').name)
    view = Path(sdf).resolve().parent / 'pose.view.mae'
    out = Path(output).with_suffix('.view.sdf')
    log = Path(output).with_suffix('.log')

    try:
        cmder.run(f'{schrodinger}/utilities/structconvert {pdb} {receptor}', exit_on_error=True)
        cmder.run(f'{schrodinger}/utilities/structconvert {sdf} {pose}', exit_on_error=True)
        cmder.run(f'cat {receptor} {pose} > {view}')

        options = ' '.join([f"-asl 'res.num {n}' -hbond {i}" for i, n in enumerate(residue_number, 1)])
        p = cmder.run(f'{schrodinger}/run pose_filter.py {view} {out} {options} -WAIT -NOJOBID > {log}',
                      fmt_cmd=False, exit_on_error=True, debug=True)
        if p.returncode == 0:
            with open(output, 'w') as o:
                f = MolIO.parse_sdf(out)
                next(f)
                for s in f:
                    o.write(s.sdf(title=f'{s.title}_{s.score}'))
    finally:
        cmder.run(f'rm -f {receptor} {pose} {view} {out} {log}', log_cmd=False)


def main():
    parser = argparse.ArgumentParser(prog='interaction-pose', description=__doc__.strip())
    parser.add_argument('sdf', help="Path to a SDF file stores docking poses")
    parser.add_argument('pdb', help="Path to a PDB file stores receptor structure")
    parser.add_argument('-r', '--residue', nargs='*', type=int,
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

    try:
        start = time.time()
        output = Path(args.output) or Path(args.sdf).resolve().parent / 'interaction.pose.sdf'

        if output.exists():
            vstool.debug_and_exit(f'Interaction pose already exists, skip re-processing\n', task=args.task, status=115)

        if args.residue:
            interaction_pose(args.sdf, args.pdb, args.residue, output=str(output), schrodinger=args.schrodinger,
                             quiet=args.quiet, verbose=args.verbose, task=args.task)
        else:
            logger.debug('No residue was provided, interaction pose filter skipped')
            cmder.run(f'cp {args.sdf} {output}')
        t = str(timedelta(seconds=time.time() - start))
        vstool.debug_and_exit(f'Filter interaction pose complete in {t.split(".")[0]}\n',
                              task=args.task, status=115)
    except Exception as e:
        vstool.error_and_exit(f'Filter interaction pose failed due to\n{e}\n\n{traceback.format_exc()}',
                              task=args.task, status=-115)


if __name__ == '__main__':
    main()
