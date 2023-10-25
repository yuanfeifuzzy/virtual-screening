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
from loguru import logger
from seqflow import Flow, task

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

parser.add_argument('--schrodinger', help='Path to Schrodinger Suite root directory', type=vstool.check_dir)

parser.add_argument('--task', type=int, default=0, help="ID associated with this task")
parser.add_argument('--cpu', type=int, default=32,
                    help="Maximum number of CPUs can be used for parallel processing, default: %(default)s")

parser.add_argument('--name', help="Name of the docking job, default: %(default)s", default='post-docking')
parser.add_argument('--partition', help='Name of the queue, default: %(default)s', default='normal')
parser.add_argument('--email', help='Email address for send status change emails')
parser.add_argument('--email-type', help='Email type for send status change emails, default: %(default)s',
                    default='ALL', choices=('NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'))
parser.add_argument('--hour', type=int, default=12, help="Number of hours each batch needs to  "
                                                         "run, default: %(default)s")
parser.add_argument('--day', type=int, default=0, help="Number of days each batch needs to  "
                                                       "run, default: %(default)s")
parser.add_argument('--dependency', help="Dependency of this task")
parser.add_argument('--submit', action='store_true', help="Submit the job to the queue for processing")
parser.add_argument('--hold', action='store_true',
                    help="Hold the submission without actually submit the job to the queue")

parser.add_argument('-q', '--quiet', help='Process data quietly without debug and info message', action='store_true')
parser.add_argument('-v', '--verbose', help='Process data verbosely with debug and info message', action='store_true')
parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
vstool.setup_logger(quiet=args.quiet, verbose=args.verbose)

outdir = args.sdf.parent
os.chdir(outdir)

setattr(args, 'outdir', outdir)
setattr(args, 'result', outdir / 'cluster.pose.sdf')
setattr(args, 'cpu', 4 if args.debug else vstool.get_available_cpus(args.cpu))


def submit():
    data = {'bin': Path(vstool.check_exe('python')).parent, 'outdir': args.outdir}
    env = ['source {bin}/activate', '', 'cd {outdir} || {{ echo "Failed to cd into {outdir}!"; exit 1; }}', '', '']

    cmd = ['post-docking', str(args.sdf), str(args.pdb),
           f'--top {args.top}', f'--clusters {args.clusters} ',
           f'--method {args.method}', f'--bits {args.bits} ',
           f'--cpu {args.cpu}', f'--task {args.task}',
           f'--schrodinger {args.schrodinger}']
    if args.residue:
        cmd.append(f'--residue {" ".join(str(x) for x in args.residue)}')

    vstool.submit('\n'.join(env).format(**data) + f' \\\n  '.join(cmd),
                  cpus_per_task=args.cpu, job_name=args.name,
                  day=args.day, hour=args.hour, partition=args.partition,
                  email=args.email, mail_type=args.email_type, log='%x.%j.log',
                  script='post.docking.sh', hold=args.hold, dependency=args.dependency)
        

def interaction_pose(sdf, out='interaction.pose.sdf'):
    if args.residue:
        logger.debug(f'Filtering pose with residue {args.residue} hydrogen bond interaction')
        cmd = (f'interaction-pose {sdf} {args.pdb} --schrodinger {args.schrodinger} '
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
    num = sum(1 for _ in MolIO.parse_sdf(sdf))
    if num <= args.clusters:
        logger.debug(f'Only {num} poses were found in {sdf}, no clustering will be performed')
        cmder.run(f'cp {sdf} {out}')
    else:
        logger.debug(f'Clustering {num:,} poses into {args.clusters:,} clusters')
        cmd = (f'cluster-pose {sdf} --clusters {args.clusters} --cpu {args.cpu} --task {args.task} '
               f'--method {args.method} --bits {args.bits} --verbose')
        p = cmder.run(cmd, exit_on_error=False, debug=True)
        if p.returncode:
            raise RuntimeError(f'Failed to run cluster-pose')


def main():
    if args.result.exists():
        logger.debug(f'Post docking analysis result {args.result} already exists, skip re-docking')
    else:
        if args.submit or args.hold or args.dependency:
            submit()
        else:
            sdf = interaction_pose(vstool.check_file(args.sdf, task=args.task))
            if sdf:
                cluster_pose(sdf)


if __name__ == '__main__':
    main()
