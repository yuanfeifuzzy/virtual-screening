#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A pipeline for perform virtual screening in an easy and smart way
"""

import os
import socket
import argparse
from pathlib import Path

import vstool

parser = argparse.ArgumentParser(prog='vs', description=__doc__.strip())
parser.add_argument('sdf', help="Path to a SDF file contains prepared ligands", type=vstool.check_file)
parser.add_argument('pdb', help="Path to a PDB file contains receptor structure", type=vstool.check_file)
parser.add_argument('--center', help="The X, Y, and Z coordinates of the center (Angstrom)",
                    nargs=3, type=float, required=True)

parser.add_argument('--flexible', help="Path to a PDBQT file contains prepared flexible receptor structure")
parser.add_argument('--size', help="The size in the X, Y, and Z dimension (Angstroms)",
                    type=int, nargs=3, default=[15, 15, 15])

parser.add_argument('--outdir', help="Path to a directory for saving output files",
                    default='.', type=vstool.mkdir)

parser.add_argument('--residue', nargs='*', type=int,
                    help="Residue numbers that interact with ligand via hydrogen bond")
parser.add_argument('--top', help="Percentage of top poses need to be retained for "
                                  "downstream analysis, default: %(default)s", type=float)
parser.add_argument('--clusters', help="Number of clusters for clustering top poses, "
                                       "default: %(default)s", type=int)
parser.add_argument('--method', help="Method for generating fingerprints",
                    choices=('morgan2', 'morgan3', 'ap', 'rdk5'))
parser.add_argument('--bits', help="Number of fingerprint bits", type=int)
parser.add_argument('--time', type=float, help="MD simulation time, default: %(default)s ns.")

parser.add_argument('--nodes', type=int, default=8, help="Number of nodes, default: %(default)s.")
parser.add_argument('--project',
                    help='The nmme of project you would like to be charged, default: %(default)s', default='MCB23087')
parser.add_argument('--email', help='Email address for send status change emails')
parser.add_argument('--email-type', help='Email type for send status change emails, default: %(default)s',
                    default='ALL', choices=('NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'))
parser.add_argument('--delay', help='Hours need to delay running the job.', type=int, default=0)
parser.add_argument('--hold', help='Only generate submit script but hold for submitting', action='store_true')

parser.add_argument('--quiet', help='Enable quiet mode.', action='store_true')
parser.add_argument('--verbose', help='Enable verbose mode.', action='store_true')
parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)

setattr(args, 'wd', vstool.mkdir(Path(os.environ.get('SCRATCH', '/scratch'), args.outdir.name)))
setattr(args, 'summary', args.outdir / 'docking.md.summary.csv')
if args.summary.exists():
    vstool.info_and_exit(f'Virtual screening result {args.summary} already exists, skip re-processing')

setattr(args, 'nodes', 1 if args.debug else args.nodes)


def task_queue():
    hostname = socket.gethostname()
    ntasks_per_node = 4 if 'frontera' in hostname else 3
    ntasks = args.nodes * ntasks_per_node
    queue = 'rtx' if 'frontera' in hostname else 'gpu-a100'
    queue = f'{queue}-dev' if args.debug else queue
    return ntasks_per_node, ntasks, queue


def main():
    ntasks_per_node, ntasks, queue = task_queue()

    source = f'source {Path(vstool.check_exe("python")).parent / "activate"}'
    cd = f'cd {args.outdir} || {{ echo "Failed to cd into {args.outdir}!"; exit 1; }}'

    (cx, cy, cz), (sx, sy, sz) = args.center, args.size
    batching = ['vs-batch-ligand', str(args.sdf), str(args.pdb), f'--batch {ntasks}',
                f'--outdir {args.wd}', f'--center {cx} {cy} {cz}', f'--size {sx} {sy} {sz}']
    if args.flexible:
        batching.append(f'--flexible {args.flexible}')
    batching = vstool.qvd(batching, args)

    launcher = 'module load launcher_gpu'
    lcmd = ['export LAUNCHER_WORKDIR={wd}',
            'export LAUNCHER_JOB_FILE={wd}/{job}.commands.txt',
            '"${{LAUNCHER_DIR}}"/paramrun', '']
    docking = '\n'.join(lcmd).format(wd=str(args.wd), job='docking')

    post_docking = ['vs-post-docking', str(args.wd), str(args.pdb), f'--outdir {args.outdir}',
                    f'--top {args.top}' if args.top else '',
                    f'--clusters {args.clusters}' if args.clusters else '',
                    f'--method {args.method}' if args.method else '',
                    f'--bits {args.bits}' if args.bits else '',
                    f'--residue {" ".join(str(x) for x in args.residue)}' if args.residue else '',
                    f'--time {args.time} --batch {ntasks}' if args.time else '']
    post_docking = vstool.qvd(post_docking, args)

    md = '\n'.join(lcmd).format(wd=str(args.wd), job='md')
    post_md = f'post-md {args.outdir} {args.summary} --scratch {args.scratch}'

    day, hour = 0 if args.debug else 1, 1 if args.debug else 23

    status, job_id = vstool.submit('\n\n'.join([source, cd, batching, launcher, docking, post_docking]),
                              nodes=args.nodes, ntasks_per_node=ntasks_per_node,
                              ntasks=ntasks, job_name='docking', day=day, hour=hour,
                              minute=59, partition=queue,
                              email=args.email, mail_type=args.email_type, log='vs.log',
                              script=args.wd / 'docking.sh', delay=args.delay,
                              project=args.project, hold=args.hold)
    
    if args.time and status == 0:
        vstool.submit('\n\n'.join([source, cd, launcher, md, post_md]),
                      nodes=args.nodes, ntasks_per_node=ntasks_per_node, ntasks=ntasks, job_name='post.docking',
                      day=day, hour=hour, partition=queue,
                      email=args.email, mail_type=args.email_type, log='vs.log', mode='append',
                      script=args.wd / 'md.sh', dependency=f'afterok:{job_id}',
                      delay=args.delay, project=args.project, hold=args.hold)


if __name__ == '__main__':
    main()
