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
parser.add_argument('receptor', help="Path to prepared rigid receptor in .pdbqt format file", type=vstool.check_file)
parser.add_argument('wd', help="Path to work directory", type=vstool.check_dir)
parser.add_argument('-o', '--outdir', default='.', help="Path to a directory for saving output files",
                    type=vstool.mkdir)
parser.add_argument('-b', '--batch', type=int, help="Number of batches that the SDF will be split to, "
                                                    "default: %(default)s", default=8)
parser.add_argument('-f', '--flexible', help="Path to prepared flexible receptor file in .pdbqt format")
parser.add_argument('-y', '--filter', help="Path to a JSON file contains descriptor filters")
parser.add_argument('-s', '--size', help="The size in the X, Y, and Z dimension (Angstroms)", type=int, nargs='+')
parser.add_argument('-c', '--center', help="T X, Y, and Z coordinates of the center", type=float, nargs='+')
parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')

args = parser.parse_args()
vstool.setup_logger(verbose=True)


def main():
    logger.debug(f'Splitting {args.ligand} into {args.batch} batches ...')
    batches = MolIO.batch_sdf(args.ligand, args.batch, args.outdir / 'batch.')

    (cx, cy, cz), (sx, sy, sz), cmds = args.center, args.size, []
    program = Path(vstool.check_exe("python")).parent / 'docking'
    with args.wd.joinpath('docking.commands.txt').open('w') as o:
        for batch in batches:
            cmd = f'{program} {batch} {args.receptor} --center {cx} {cy} {cz} --size {sx} {sy} {sz}'
            if args.flexible:
                cmd = f'{cmd} --flexible {args.flexible}'
            if args.filter:
                cmd = f'{cmd} --filter {args.filter}'
            if args.debug:
                cmd = f'{cmd} --debug'
            cmds.append(cmd)
        o.write('\n'.join(cmds))


if __name__ == '__main__':
    main()
