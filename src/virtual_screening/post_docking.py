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
parser.add_argument('--schrodinger', help='Path to Schrodinger Suite root directory, default: %(default)s',
                        type=vstool.check_dir, default='/work/08944/fuzzy/share/software/DESRES/2023.2')

parser.add_argument('--md', help='Path to md executable, default: %(default)s',
                    type=vstool.check_exe,
                    default='/work/08944/fuzzy/share/software/virtual-screening/venv/lib/python3.11/site-packages/virtual_screening/desmond_md.sh')
parser.add_argument('--time', type=float, default=50, help="MD simulation time, default: %(default)s ns.")
parser.add_argument('--summary', help='Path to a CSV file for saving MD summary results.')

parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
vstool.setup_logger(verbose=True)


def concatenate_sdf():
    sdf = 'docking.sdf'
    cmder.run(f'cat {args.wd}/*.docking.sdf > {sdf}')
    if not args.debug:
        cmder.run(f'rm {args.wd}/*.docking.sdf')
    cmder.run(f'cp {sdf} > {Path(args.summary).parent / sdf}')
    return sdf


def interaction_pose(sdf, out='interaction.pose.sdf'):
    if args.residue:
        logger.debug(f'Filtering pose with residue {args.residue} hydrogen bond interaction')
        cmd = (f'interaction-pose {sdf} {args.pdb} --schrodinger {args.schrodinger} '
               f'--residue {" ".join(str(x) for x in args.residue)} --output {out}')
        cmder.run(cmd, fmt_cmd=False, debug=True)
    else:
        logger.debug(f'No residue was provided, retrieving top {args.top} percent poses')
        num = sum(1 for _ in MolIO.parse_sdf(sdf))
        n = int(num * args.top / 100)
        with open(out, 'w') as o:
            for i, s in enumerate(MolIO.parse_sdf(sdf)):
                if i == n:
                    break
                o.write(s.sdf(title=f'{s.title}_{s.score}'))
    cmder.run(f'cp {out} {Path(args.summary).parent / out}')
    return out


def cluster_pose(sdf):
    out, md, mds = 'cluster.pose.sdf', 'md.commands.txt', []

    num = sum(1 for _ in MolIO.parse_sdf(sdf))
    if num <= args.clusters:
        logger.debug(f'Only {num} poses were found in {sdf}, no clustering will be performed')
        cmder.run(f'cp {sdf} {out}')
    else:
        logger.debug(f'Clustering {num:,} poses into {args.clusters:,} clusters')
        cmd = (f'cluster-pose {sdf} --clusters {args.clusters} '
               f'--method {args.method} --bits {args.bits} --output {out}')
        cmder.run(cmd, debug=True)

    cmder.run(f'cp {out} {Path(args.summary).parent / out}')

    wd = vstool.mkdir('md')
    program = Path(vstool.check_exe("python")).parent / 'molecule-dynamics'

    for s in MolIO.parse_sdf(out):
        if s.mol:
            output = s.sdf(output=wd / f'{s.title}.sdf')
            cmd = f'{program} {output} {args.pdb} --time {args.time} --exe {args.md}'
            if args.summary:
                cmd = f'{cmd} --summary {args.summary}'
            if args.debug:
                cmd = f'{cmd} --debug'
            mds.append(cmd)

    if mds and args.time:
        with md.open('w') as o:
            o.write(f'{x}\n' for x in mds)


def main():
    os.chdir(args.wd)
    sdf = concatenate_sdf()
    sdf = interaction_pose(sdf)
    cluster_pose(sdf)


if __name__ == '__main__':
    main()
