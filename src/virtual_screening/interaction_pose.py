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
from rdkit import Chem
from pandarallel import pandarallel

parser = argparse.ArgumentParser(prog='interaction-pose', description=__doc__.strip())
parser.add_argument('sdf', help="Path to a SDF file stores docking poses", type=vstool.check_file)
parser.add_argument('pdb', help="Path to a PDB file stores receptor structure", type=vstool.check_file)
parser.add_argument('--residue', nargs='*', type=int,
                    help="Residue numbers that interact with ligand via hydrogen bond")
parser.add_argument('--output', help="Path to a SDF file for saving output poses",
                    default='interaction.pose.sdf')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)
setattr(args, 'output', args.output or args.sdf.parent / 'interaction.pose.sdf')


def main():
    if Path(args.output).exists():
        logger.debug(f'Interaction pose filter result {args.output} already exists, skip re-filtering')
    else:
        os.chdir(args.sdf.parent)
        if args.residue:
            receptor, pose, view, out, log = 'receptor.mae', 'pose.mae', 'view.mae', 'view.sdf', 'interaction.pose.log'
            run = '/work/02940/ztan818/ls6/software/DESRES/2023.2/run'
            structconvert = '/work/02940/ztan818/ls6/software/DESRES/2023.2/utilities/structconvert'
            try:
                cmder.run(f'{structconvert} {args.pdb} {receptor}', exit_on_error=True)
                cmder.run(f'{structconvert} {args.sdf} {pose}', exit_on_error=True)
                cmder.run(f'cat {receptor} {pose} > {view}', exit_on_error=True)

                options = ' '.join([f"-asl 'res.num {n}' -hbond {i}" for i, n in enumerate(args.residue, 1)])
                p = cmder.run(f'{run} pose_filter.py {view} {out} {options} -WAIT -NOJOBID > {log}',
                              fmt_cmd=False, exit_on_error=True, debug=True)
                if p.returncode == 0:
                    with open(args.output, 'w') as o:
                        f = MolIO.parse_sdf(out)
                        next(f)
                        for s in f:
                            o.write(s.sdf(title=f'{s.title}_{s.score}'))
            finally:
                cmder.run(f'rm -f {pose} {view} {out} {log}', log_cmd=False)
        else:
            logger.debug('No residue was provided, interaction pose filter skipped')
            with open(args.output, 'w') as o:
                f = MolIO.parse_sdf(args.sdf)
                for s in f:
                    o.write(s.sdf(title=f'{s.title}_{s.score}'))


if __name__ == '__main__':
    main()
