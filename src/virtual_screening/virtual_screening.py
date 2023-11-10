#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A pipeline for perform virtual screening in an easy and smart way
"""

import os
import socket
import argparse
from pathlib import Path

import cmder
import vstool

parser = argparse.ArgumentParser(prog='virtual-screening', description=__doc__.strip())
parser.add_argument('ligand', help="Path to a directory contains prepared ligands in SDF format",
                    type=vstool.check_file)
parser.add_argument('pdb', help="Path to the receptor in PDB format file", type=vstool.check_file)
parser.add_argument('center', help="The X, Y, and Z coordinates of the center", type=float, nargs='+')

parser.add_argument('--flexible', help="Path to prepared flexible receptor file in PDBQT format")
parser.add_argument('--filter', help="Path to a JSON file contains descriptor filters")
parser.add_argument('--size', help="The size in the X, Y, and Z dimension (Angstroms)",
                    type=int, nargs='*', default=[15, 15, 15])

parser.add_argument('--outdir', help="Path to a directory for saving output files", type=vstool.mkdir)
parser.add_argument('--scratch', help="Path to the scratch directory, default: %(default)s",
                    default=Path(os.environ.get('SCRATCH', '/scratch')))

parser.add_argument('--residue', nargs='*', type=int,
                    help="Residue numbers that interact with ligand via hydrogen bond")
parser.add_argument('--top', help="Percentage of top poses need to be retained for "
                                  "downstream analysis, default: %(default)s", type=float, default=10)
parser.add_argument('--clusters', help="Number of clusters for clustering top poses, "
                                       "default: %(default)s", type=int, default=100)
parser.add_argument('--method', help="Method for generating fingerprints, default: %(default)s",
                    default='morgan2', choices=('morgan2', 'morgan3', 'ap', 'rdk5'))
parser.add_argument('--bits', help="Number of fingerprint bits, default: %(default)s", default=1024, type=int)
parser.add_argument('--time', type=float, default=50, help="MD simulation time, default: %(default)s ns.")
parser.add_argument('--batch', type=int, help="Number of batches that the ligands will be split to, "
                                                    "default: %(default)s", default=0)

parser.add_argument('--nodes', type=int, default=8, help="Number of nodes, default: %(default)s.")
parser.add_argument('--project', help='The nmme of project you would like to be charged')
parser.add_argument('--email', help='Email address for send status change emails')
parser.add_argument('--email-type', help='Email type for send status change emails, default: %(default)s',
                    default='ALL', choices=('NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'))
parser.add_argument('--delay', help='Hours need to delay running the job.', type=int, default=0)
parser.add_argument('--hold', help='Only generate submit script but hold for submitting', action='store_true')

parser.add_argument('--docking-only', help='Only perform docking and post-docking and no MD.', action='store_true')
parser.add_argument('--separate', help='Separate docking and MD into 2 steps', action='store_true')
parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)

setattr(args, 'summary', args.outdir / 'docking.md.summary.csv')
if args.summary.exists():
    vstool.debug_and_exit(f'Virtual screening result {args.summary} already exists, skip re-processing')

setattr(args, 'scratch', vstool.mkdir(args.scratch / f'{args.outdir.name}'))
setattr(args, 'nodes', 1 if args.debug else args.nodes)
setattr(args, 'batch', 400)


def str_cmd(cmd):
    if args.debug:
        cmd.append('--debug')
    return ' \\\n  '.join(cmd)


def main():
    hostname = socket.gethostname()
    ntasks_per_node = 4 if 'frontera' in hostname else 3
    ntasks = args.nodes * ntasks_per_node
    queue = 'rtx' if 'frontera' in hostname else 'gpu-a100'
    queue = f'{queue}-dev' if args.debug else queue
    device = ','.join([str(x) for x in range(ntasks_per_node)] * args.nodes)

    source = f'source {Path(vstool.check_exe("python")).parent / "activate"}'
    cd = f'cd {args.outdir} || {{ echo "Failed to cd into {args.outdir}!"; exit 1; }}'

    (cx, cy, cz), (sx, sy, sz) = args.center, args.size
    batching = ['batch-ligand', str(args.ligand), str(args.pdb),
           f'--outdir {args.scratch}', f'--batch {args.batch}',
           f'--center {cx} {cy} {cz}', f'--size {sx} {sy} {sz}']
    if args.filter:
        batching.append(f'--filter {args.filter}')
    if args.flexible:
        batching.append(f'--flexible {args.flexible}')
    batching = str_cmd(batching)

    launcher = 'module load launcher_gpu'
    lcmd = ['export LAUNCHER_WORKDIR={outdir}',
            'export LAUNCHER_JOB_FILE={outdir}/{job}.commands.txt',
            '"${{LAUNCHER_DIR}}"/paramrun', '']
    docking = '\n'.join(lcmd).format(outdir=str(args.scratch), job='docking')

    post_docking = ['post-docking', str(args.scratch), str(args.pdb), f'--top {args.top}', 
                    f'--clusters {args.clusters} ', f'--method {args.method}', f'--bits {args.bits} ', 
                    f'--time {args.time}', f'--outdir {args.outdir}',
                    f'--batch {ntasks}', f'--device {device}']
    if args.residue:
        post_docking.append(f'--residue {" ".join(str(x) for x in args.residue)}')
    post_docking = str_cmd(post_docking)
    
    md = '\n'.join(lcmd).format(outdir=str(args.scratch), job='md')
    post_md =f'post-md {args.outdir} {args.summary.name} --scratch {args.scratch}'
    
    if args.docking_only:
        vstool.submit('\n\n'.join([source, cd, batching, launcher, docking, post_docking]),
                      nodes=args.nodes, ntasks_per_node=ntasks_per_node, ntasks=ntasks, job_name='docking',
                      hour=1 if args.debug else 11, minute=59, partition=queue,
                      email=args.email, mail_type=args.email_type, log='docking.log',
                      script=args.outdir / 'docking.sh', delay=args.delay,
                      project=args.project, hold=args.hold)
    else:
        if args.separate:
            _, job_id = vstool.submit('\n\n'.join([source, cd, batching, launcher, docking, post_docking]),
                                      nodes=args.nodes, ntasks_per_node=ntasks_per_node,
                                      ntasks=ntasks, job_name='docking',
                                      hour=1 if args.debug else 11, minute=59, partition=queue,
                                      email=args.email, mail_type=args.email_type, log='vs.log',
                                      script=args.outdir / 'docking.sh', delay=args.delay,
                                      project=args.project, hold=args.hold)

            vstool.submit('\n\n'.join([launcher, md, post_md]),
                          nodes=args.nodes, ntasks_per_node=ntasks_per_node, ntasks=ntasks, job_name='post.docking',
                          day=0 if args.debug else 1, hour=1 if args.debug else 23, minute=59, partition=queue,
                          email=args.email, mail_type=args.email_type, log='vs.log', mode='append',
                          script=args.outdir / 'md.sh', dependency=f'afterok:{job_id}',
                          delay=args.delay, project=args.project, hold=args.hold)
        else:
            vstool.submit('\n\n'.join([source, cd, batching, launcher, docking, post_docking, md, post_md]),
                          nodes=args.nodes, ntasks_per_node=ntasks_per_node,
                          ntasks=ntasks, job_name='vs',
                          day=0 if args.debug else 1, hour=1 if args.debug else 23, minute=59, partition=queue,
                          email=args.email, mail_type=args.email_type, log='vs.log',
                          script=args.outdir / 'vs.sh', delay=args.delay, project=args.project, hold=args.hold)
    

if __name__ == '__main__':
    main()
