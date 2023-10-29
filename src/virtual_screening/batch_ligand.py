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
                    type=vstool.mkdir, required=True)
parser.add_argument('-b', '--batch', type=int, help="Number of batches that the SDF will be split to, "
                                                    "default: %(default)s", default=8, required=True)
parser.add_argument('-c', '--center', help="The X, Y, and Z coordinates of the center",
                    type=float, nargs='+', required=True)

parser.add_argument('-f', '--flexible', help="Path to prepared flexible receptor file in .pdbqt format")
parser.add_argument('-y', '--filter', help="Path to a JSON file contains descriptor filters")
parser.add_argument('--size', help="The size in the X, Y, and Z dimension (Angstroms)",
                    type=int, nargs='*', default=[15, 15, 15])

parser.add_argument('--pdb', help="Path to a PDB file contains receptor structure", type=vstool.check_file)
parser.add_argument('--residue', nargs='*', type=int,
                    help="Residue numbers that interact with ligand via hydrogen bond")
parser.add_argument('--top', help="Percentage of top poses need to be retained for "
                                  "downstream analysis, default: %(default)s", type=float, default=10)
parser.add_argument('--clusters', help="Number of clusters for clustering top poses, "
                                       "default: %(default)s", type=int, default=1000)
parser.add_argument('--method', help="Method for generating fingerprints, default: %(default)s",
                    default='morgan2', choices=('morgan2', 'morgan3', 'ap', 'rdk5'))
parser.add_argument('--bits', help="Number of fingerprint bits, default: %(default)s", default=1024, type=int)
parser.add_argument('--schrodinger', help='Path to Schrodinger Suite root directory, default: %(default)s',
                        type=vstool.check_dir, default='/work/08944/fuzzy/share/software/DESRES/2023.2')
parser.add_argument('--summary', help='Path to a CSV file for saving MD summary results.')

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
            
            if args.pdb:
                cmd = (f'{cmd} --pdb {args.pdb} --top {args.top} --cluster {args.cluster} '
                       f'--method {args.method} --bits {args.bits} --schrodinger {args.schrodinger}')
                if args.residue:
                    cmd = f'{cmd} --residue {" ".join(str(x) for x in residue)}'
                if args.summary:
                    cmd = f'{cmd} --summary {args.summary}'
                    
            if args.debug:
                cmd = f'{cmd} --debug'
            cmds.append(cmd)
        o.write('\n'.join(cmds))


if __name__ == '__main__':
    main()
