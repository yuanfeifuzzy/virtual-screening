#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare receptor to generate .pdbqt and field grid files
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import timedelta

import cmder
import vstool

parser = argparse.ArgumentParser(prog='vs-prepare-receptor', description=__doc__.strip())
parser.add_argument('pdb', help="Path to the receptor in PDB format file", type=vstool.check_file)
parser.add_argument('--center', help="The X, Y, and Z coordinates of the center",
                    type=float, nargs='+', required=True)
parser.add_argument('--size', help="The size in the X, Y, and Z dimension (Angstroms)",
                    type=int, nargs='*', default=[15, 15, 15])
parser.add_argument('--outdir', help="Path to a directory for saving output files", type=vstool.mkdir)
parser.add_argument('--autodock4', help="Path to AutdoDock 4 installation directory", type=vstool.mkdir)
parser.add_argument('--mgltools', help="Path to MGLTools installation directory", type=vstool.mkdir)
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)

def main():
    mgltools = args.mgltools or '/software/MGLTools'
    autodock4 = args.autdock4 or '/software/AutoDock/4.2.6/'

    __python__ = f'{mgltools}/bin/pythonsh'
    __autodocktools__ = f'{mgltools}/MGLToolsPckgs/AutoDockTools/Utilities24'
    prepare_receptor4 = f'{__python__} {__autodocktools__}/prepare_receptor4.py'
    prepare_gpf4 = f'{__python__} {__autodocktools__}/prepare_gpf4.py'
    autogrid4 = f'{autodock4}/autogrid4'

    wd, pdb, name = args.outdir, args.pdb, args.pdb.with_suffix('')
    pdbqt, gpf = f'{name}.pdbqt', f'{name}.gpf'
    cmder.run(f'cp {args.pdb} {args.pdb.name}', cwd=str(wd), exit_on_error=True)
    cmder.run(f'{prepare_receptor4} -r {pdb} -A hydrogens -U waters', cwd=str(wd), exit_on_error=True)

    ligand_types = "-p ligand_types='A,C,HD,N,NA,OA,SA,S,P,Br,F,Cl'"
    (cx, cy, cz), (sx, sy, sz) = args.center, args.size
    grid_center = f"-p gridcenter='{cx},{cy},{cz}'"
    x, y, z = int(sx / 0.375), int(sy / 0.375), int(sz / 0.375)
    cmd = f"{prepare_gpf4} -r {pdbqt} {ligand_types} {grid_center} -p npts='{x},{y},{z}'"

    cmder.run(cmd, cwd=str(wd), exit_on_error=True, fmt_cmd=False)
    cmder.run(f'{autogrid4} -p {gpf} &> /dev/null', cwd=str(wd), exit_on_error=True)


if __name__ == '__main__':
    main()
