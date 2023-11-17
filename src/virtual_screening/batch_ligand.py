#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A command line tool for easily generating ligand batches
"""

import json
import os
import sys
import argparse
import traceback
from pathlib import Path

import cmder
import MolIO
import vstool
from loguru import logger

parser = argparse.ArgumentParser(prog='batch-ligand', description=__doc__.strip())
parser.add_argument('ligand', help="Path to a single SDF file contains prepared ligands", type=vstool.check_file)
parser.add_argument('pdb', help="Path to the receptor in PDB format file", type=vstool.check_file)
parser.add_argument('--outdir',help="Path to a directory for saving output files", type=vstool.mkdir)
parser.add_argument('-b', '--batch', type=int, help="Number of batches that the SDF will be split to, "
                                                    "default: %(default)s", default=8, required=True)
parser.add_argument('--center', help="The X, Y, and Z coordinates of the center",
                    type=float, nargs='+', required=True)

parser.add_argument('--flexible', help="Path to prepared flexible receptor file in .pdbqt format")
parser.add_argument('--filter', help="Path to a JSON file contains descriptor filters")
parser.add_argument('--size', help="The size in the X, Y, and Z dimension (Angstroms)",
                    type=int, nargs='*', default=[15, 15, 15])

parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')

args = parser.parse_args()
vstool.setup_logger(verbose=True)


def main():
    logger.debug(f'Splitting {args.ligand} into batches ...')
    batches = MolIO.batch_sdf(args.ligand, args.batch, args.outdir / 'batch.')
    logger.debug(f'Successfully split {args.ligand} into {len(batches)} batches ...')

    (cx, cy, cz), (sx, sy, sz), cmds = args.center, args.size, []
    center, size = f'--center {cx} {cy} {cz}', f'--size {sx} {sy} {sz}'
    program = Path(vstool.check_exe("python")).parent / 'docking'
    pdbqt = args.outdir / 'receptor' / f'{args.pdb.name}qt'
    if not pdbqt.exists():
        cmder.run(f'receptor-preparation {args.pdb} {center} {size} --outdir {pdbqt.parent}', exit_on_error=True)

    output = args.outdir.joinpath('docking.commands.txt')
    with output.open('w') as o:
        for batch in batches:
            cmd = f'{program} {batch} {pdbqt} {center} {size} --outdir {args.outdir}'
            if args.flexible:
                cmd = f'{cmd} --flexible {args.flexible}'
            if args.filter:
                cmd = f'{cmd} --filter {args.filter}'
            if args.debug:
                cmd = f'{cmd} --debug'
            cmds.append(cmd)
        cmds = cmds[:8] if args.debug else cmds
        o.writelines(f'{cmd}\n' for cmd in cmds)
    logger.debug(f'Successfully saved the following {len(cmds)} docking commands into {output}')


if __name__ == '__main__':
    main()
