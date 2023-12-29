#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A command line tool for pst-docking analysis
"""

import os
import argparse
from pathlib import Path

import cmder
import MolIO
import vstool

parser = argparse.ArgumentParser(prog='vs-post-docking', description=__doc__.strip())
parser.add_argument('wd', help="Path to a directory contains docking output", type=vstool.check_dir)
parser.add_argument('pdb', help="Path to a PDB file contains receptor structure", type=vstool.check_file)

parser.add_argument('--residue', nargs='*', type=int,
                    help="Residue numbers that interact with ligand via hydrogen bond")
parser.add_argument('--top', help="Percentage of top poses need to be retained for "
                                  "downstream analysis, default: %(default)s", type=float, default=10)
parser.add_argument('--clusters', help="Number of clusters for clustering top poses", type=int)
parser.add_argument('--method', help="Method for generating fingerprints, default: %(default)s",
                    default='morgan2', choices=('morgan2', 'morgan3', 'ap', 'rdk5'))
parser.add_argument('--bits', help="Number of fingerprint bits, default: %(default)s", default=1024, type=int)

parser.add_argument('--time', type=float, help="MD simulation time, default: %(default)s ns.")
parser.add_argument('--outdir', help="Path to a directory for saving output files",
                    type=vstool.mkdir, default='.')
parser.add_argument('--batches', type=int, help="Number of batches that the SDF for MD will be split to, "
                                                    "default: %(default)s", default=0)

parser.add_argument('--quiet', help='Enable quiet mode.', action='store_true')
parser.add_argument('--verbose', help='Enable verbose mode.', action='store_true')
parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)

setattr(args, 'scratch', vstool.mkdir(f"{os.environ.get('SCRATCH', '/scratch')}/{args.outdir.name}"))


def main():
    sdf = MolIO.merge_sdf(args.wd.glob('*.docking.sdf.gz'), args.outdir / 'docking.sdf.gz')
    if not args.debug:
        cmder.run(f'rm {args.wd}/*docking.sdf.gz')
        
    if args.residue or args.clusters:
        interaction_pose = args.outdir / 'interaction.pose.sdf'
        cmd = f'interaction-pose {sdf} {args.pdb} --output {interaction_pose}'
        if args.residue:
            cmd = f'{cmd} --residue {" ".join(str(x) for x in args.residue)}'
        cmder.run(cmd, exit_on_error=True, debug=True)

        cluster_pose = args.outdir / 'cluster.pose.sdf'
        cmd = (f'cluster-pose {interaction_pose} --clusters {args.clusters} --method {args.method} --bits {args.bits} '
               f'--top {args.top} --output {cluster_pose}')
        cmder.run(cmd, exit_on_error=True, debug=True)

        if args.time:
            wd, mae, commands = vstool.mkdir(args.wd / 'md'), args.wd / 'receptor.mae', args.wd / 'md.commands.txt'

            if args.batch:
                sdfs = MolIO.batch_sdf(cluster_pose, args.batches, str(args.scratch / 'cluster.pose.'))
            else:
                sdfs = [s.sdf(output=args.scratch / f'{s.title}.sdf') for s in MolIO.parse_sdf(cluster_pose) if s.mol]

            with open(commands, 'w') as o:
                for sdf in sdfs:
                    cmd = ['vs-md', sdf, args.pdb, f'--time {int(1000*args.time)}', f'--outdir {args.outdir}']
                    o.write(f'{vstool.qvd(cmd, args, sep=" ")}\n')
            logger.debug(f'Successfully saved {len(sdfs)} md launch commands to {commands}')


if __name__ == '__main__':
    main()
