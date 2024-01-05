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

import cmder
import MolIO
import vstool


parser = argparse.ArgumentParser(prog='post-docking', description=__doc__.strip())
parser.add_argument('sdf', help="Path to a SDF file contains docking poses")
parser.add_argument('pdb', help="Path to a PDB file contains receptor structure", type=vstool.check_file)

parser.add_argument('-r', '--residue', nargs='*', type=int,
                        help="Residue numbers that interact with ligand via hydrogen bond")
parser.add_argument('-t', '--top', help="Percentage of top poses need to be retained for "
                                                "downstream analysis, default: %(default)s", type=float, default=10)
parser.add_argument('-c', '--clusters', help="Number of clusters for clustering top poses, "
                                                "default: %(default)s", type=int, default=1000)
parser.add_argument('-m', '--method', help="Method for generating fingerprints, default: %(default)s",
                    default='morgan2', choices=('morgan2', 'morgan3', 'ap', 'rdk5'))
parser.add_argument('-b', '--bits', help="Number of fingerprint bits, default: %(default)s", default=1024, type=int)

parser.add_argument('--schrodinger', help='Path to Schrodinger Suite root directory, default: %(default)s',
                        type=vstool.check_dir, default='/software/SchrodingerSuites/2022.4')

parser.add_argument('--task', type=int, default=0, help="ID associated with this task")
parser.add_argument('--cpu', type=int, default=32,
                    help="Maximum number of CPUs can be used for parallel processing, default: %(default)s")

parser.add_argument('-q', '--quiet', help='Process data quietly without debug and info message', action='store_true')
parser.add_argument('-v', '--verbose', help='Process data verbosely with debug and info message', action='store_true')
parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(quiet=args.quiet, verbose=args.verbose)

outdir = Path(args.sdf).parent
os.chdir(outdir)

setattr(args, 'outdir', outdir)
setattr(args, 'result', outdir / 'cluster.pose.sdf')


def interaction_pose(sdf, out='interaction.pose.sdf'):
    if Path(out).exists():
        logger.debug(f'Output {out} already exists, skip re-processing')
    else:
        if args.residue:
            logger.debug(f'Filtering pose with residue {args.residue} hydrogen bond interaction')
            cmd = (f'vs-interaction-pose {sdf} {args.pdb} --schrodinger {args.schrodinger} '
                   f'--residue {" ".join(str(x) for x in args.residue)} '
                   f'--output {out} --task {args.task} --verbose')
            p = cmder.run(cmd, fmt_cmd=False, exit_on_error=False, debug=True)
            if p.returncode:
                out = ''
        else:
            logger.debug(f'No residue was provided, retrieving top {args.top} percent poses')
            num = sum(1 for _ in MolIO.parse_sdf(sdf))
            n = int(num * args.top / 100)
            with open(out, 'w') as o:
                for i, s in enumerate(MolIO.parse_sdf(sdf)):
                    if i == n:
                        break
                    o.write(s.sdf(title=f'{s.title}_{s.score}'))
    return out
                

def cluster_pose(sdf, out=args.result.name):
    if Path(out).exists():
        logger.debug(f'Output {out} already exists, skip re-processing')
    else:
        num = sum(1 for _ in MolIO.parse_sdf(sdf))
        if num <= args.clusters:
            logger.debug(f'Only {num} poses were found in {sdf}, no clustering will be performed')
            cmder.run(f'cp {sdf} {out}')
        else:
            logger.debug(f'Clustering {num:,} poses into {args.clusters:,} clusters')
            cmd = (f'vs-cluster-pose {sdf} --clusters {args.clusters} --cpu {args.cpu} --task {args.task} '
                   f'--method {args.method} --bits {args.bits} --verbose')
            p = cmder.run(cmd, exit_on_error=False, debug=True)
            if p.returncode:
                raise RuntimeError(f'Failed to run cluster-pose')


def main():
    if args.result.exists():
        logger.debug(f'Post docking analysis result {args.result} already exists, skip re-docking')
    else:
        sdf = interaction_pose(vstool.check_file(args.sdf, task=args.task))
        if sdf:
            cluster_pose(sdf)


if __name__ == '__main__':
    main()
