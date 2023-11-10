#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A command line tool for pst-docking analysis
"""

import json
import os
import sys
import argparse
import traceback
from pathlib import Path
from multiprocessing import cpu_count

import cmder
import MolIO
import vstool
from loguru import logger

parser = argparse.ArgumentParser(prog='post-docking', description=__doc__.strip())
parser.add_argument('wd', help="Path to a directory contains docking output", type=vstool.check_dir)
parser.add_argument('pdb', help="Path to a PDB file contains receptor structure", type=vstool.check_file)

parser.add_argument('--residue', nargs='*', type=int,
                    help="Residue numbers that interact with ligand via hydrogen bond")
parser.add_argument('--top', help="Percentage of top poses need to be retained for "
                                  "downstream analysis, default: %(default)s", type=float, default=10)
parser.add_argument('--clusters', help="Number of clusters for clustering top poses, "
                                       "default: %(default)s", type=int, default=1000)
parser.add_argument('--method', help="Method for generating fingerprints, default: %(default)s",
                    default='morgan2', choices=('morgan2', 'morgan3', 'ap', 'rdk5'))
parser.add_argument('--bits', help="Number of fingerprint bits, default: %(default)s", default=1024, type=int)

parser.add_argument('--time', type=float, default=50, help="MD simulation time, default: %(default)s ns.")
parser.add_argument('--outdir', help="Path to a directory for saving output files", type=vstool.mkdir)
parser.add_argument('--batch', type=int, help="Number of batches that the SDF for MD will be split to, "
                                                    "default: %(default)s", default=0)
parser.add_argument('--device', help='Comma separated list of GPU device indices')
parser.add_argument('--docking-only', help='Only perform docking and post-docking and no MD.', action='store_true')

parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
vstool.setup_logger(verbose=True)


def concatenate_sdf():
    sdf = args.wd / 'docking.sdf'
    cmder.run(f'cat {args.wd}/*.docking.sdf > {sdf}')
    if not args.debug:
        cmder.run(f'rm {args.wd}/*.docking.sdf')
    cmder.run(f'cp {sdf} {args.outdir / sdf.name}')
    return sdf


def main():
    sdf, root = concatenate_sdf(), Path(vstool.check_exe("python")).parent
    if args.docking_only:
        logger.debug(f'Docking task complete, results were successfully saved to {sdf}\n')
    else:
        structconvert = '/work/02940/ztan818/ls6/software/DESRES/2023.2/utilities/structconvert'

        interaction_pose = args.wd / 'interaction.pose.sdf'
        cmd = f'interaction-pose {sdf} {args.pdb} --output {interaction_pose}'
        if args.residue:
            cmd = f'{cmd} --residue {" ".join(str(x) for x in args.residue)}'
        cmder.run(cmd, exit_on_error=True, debug=True)

        cluster_pose = args.wd / 'cluster.pose.sdf'
        cmd = (f'cluster-pose {interaction_pose} --clusters {args.clusters} --method {args.method} --bits {args.bits} '
               f'--top {args.top} --output {cluster_pose}')
        cmder.run(cmd, exit_on_error=True, debug=True)
        cmder.run(f'cp {cluster_pose} {args.outdir}/')

        summary, md = args.outdir / 'docking.md.summary.csv', root / 'molecule-dynamics'
        wd, mae, commands = vstool.mkdir(args.wd / 'md'), args.wd / 'receptor.mae', args.wd / 'md.commands.txt'

        if not mae.exists():
            cmder.run(f'{structconvert} {args.pdb} {mae}', exit_on_error=True)

        if args.batch:
            sdfs = MolIO.batch_sdf(str(cluster_pose), args.batch, str(args.wd / 'cluster.pose.'))
            devices = args.device.split(',')
        else:
            sdfs = [s.sdf(output=wd / f'{s.title}.sdf') for s in MolIO.parse_sdf(cluster_pose) if s.mol]
            devices = ['0'] * len(sdfs)

        with open(commands, 'w') as o:
            for sdf, device in zip(sdfs, devices):
                cuda = f'export CUDA_VISIBLE_DEVICES={device}'
                cmd = f'{cuda} && {md} {sdf} {mae} --time {int(1000*args.time)} --outdir {args.outdir}'
                if args.debug:
                    cmd = f'{cmd} --debug'
                o.write(f'{cmd}\n')
        logger.debug(f'Successfully saved {len(sdfs)} md launch commands to {commands}')
    

if __name__ == '__main__':
    main()
