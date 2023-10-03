#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare ligands in variety file formats into 3-d coordinates in (sdf or pdbqt format), ready for molecular docking
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import timedelta

import cmder
import utility
from svs import tools


def main():
    parser = argparse.ArgumentParser(prog='ligand-preparation', description=__doc__.strip())
    parser.add_argument('pdb', help="Path to a PDB file contains structure of the receptor")
    parser.add_argument('center_x', help="X coordinate of the center", type=float)
    parser.add_argument('center_y', help="Y coordinate of the center", type=float)
    parser.add_argument('center_z', help="Z coordinate of the center", type=float)
    parser.add_argument('-x', help="The size in the X dimension (Angstroms)", type=float)
    parser.add_argument('-y', help="The size in the X dimension (Angstroms)", type=float)
    parser.add_argument('-z', help="The size in the X dimension (Angstroms)", type=float)
    parser.add_argument('--task', type=int, default=0, help="ID associated with this task")
    parser.add_argument('--mgltools', help="Path to the MGLTools root directory")
    parser.add_argument('--autodock4', help="Path to the AutoDock4 root directory")
    parser.add_argument('-q', '--quiet', help='Process data quietly without debug and info message',
                        action='store_true')
    parser.add_argument('-v', '--verbose', help='Process data verbosely with debug and info message',
                        action='store_true')
    parser.add_argument('--version', version=utility.get_version(__package__), action='version')

    args = parser.parse_args()

    try:
        utility.setup_logger(quiet=args.quiet, verbose=args.verbose)

        start = time.time()
        mgltools, autodock4 = Path(args.mgltools), Path(args.autodock4)

        __python__ = mgltools / 'bin/pythonsh'
        __autodocktools__ = mgltools / 'MGLToolsPckgs/AutoDockTools/Utilities24'
        prepare_receptor4 = f'{__python__} {__autodocktools__}/prepare_receptor4.py'
        prepare_gpf4 = f'{__python__} {__autodocktools__}/prepare_gpf4.py'
        autogrid4 = f'{autodock4}/autogrid4'

        wd, pdb = Path(args.pdb).parent, Path(args.pdb).name
        slug = Path(pdb).with_suffix('')
        pdbqt, gpf = f'{slug}.pdbqt', f'{slug}.gpf'
        p = cmder.run(f'{prepare_receptor4} -r {pdb} -A hydrogens -U waters', cwd=str(wd), exit_on_error=False)
        if p.returncode:
            utility.error_and_exit('Failed to make .pdbqt file for receptor', task=args.task, status=60)

        ligand_types = "-p ligand_types='A,C,HD,N,NA,OA,SA,S,P,Br,F,Cl'"
        grid_center = f"-p gridcenter='{args.center_x},{args.center_y},{args.center_z}'"
        cmd = f'{prepare_gpf4} -r {pdbqt} {ligand_types} {grid_center}'
        if all(x is not None for x in (args.x, args.y, args.z)):
            cmd = f"{cmd} -p npts='{args.x},{args.y},{args.z}'"
        p = cmder.run(cmd, cwd=str(wd), exit_on_error=False, fmt_cmd=False)
        if p.returncode:
            utility.error_and_exit('Failed to make .gpf file for receptor', task=args.task, status=60)

        p = cmder.run(f'{autogrid4} -p {gpf} &> /dev/null', cwd=str(wd), exit_on_error=False)
        if p.returncode:
            utility.error_and_exit('Failed to make maps grid files for receptor', task=args.task, status=60)

        t = str(timedelta(seconds=time.time() - start))
        utility.debug_and_exit(f'Receptor preparation complete in {t.split(".")[0]}\n', task=args.task, status=60)
    except Exception as e:
        utility.error_and_exit(f'Ligand preparation failed due to:\n{e}\n\n{traceback.format_exc()}',
                               task=args.task, status=-60)


if __name__ == '__main__':
    main()
