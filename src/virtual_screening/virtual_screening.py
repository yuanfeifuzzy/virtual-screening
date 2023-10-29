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


def cmding(env, cmd, args):
    if args.debug:
        cmd.append('--debug')
    return env + ' \\\n  '.join(cmd)


def main():
    parser = argparse.ArgumentParser(prog='virtual-screening', description=__doc__.strip())
    parser.add_argument('ligand', help="Path to a directory contains prepared ligands in SDF format",
                        type=vstool.check_file)
    parser.add_argument('receptor', help="Path to prepared rigid receptor in PDBQT format file",
                        type=vstool.check_file)
    parser.add_argument('center', help="The X, Y, and Z coordinates of the center", type=float, nargs='+')

    parser.add_argument('--pdb', help="Path to prepared receptor structure in PDB format")
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
                                           "default: %(default)s", type=int, default=1000)
    parser.add_argument('--method', help="Method for generating fingerprints, default: %(default)s",
                        default='morgan2', choices=('morgan2', 'morgan3', 'ap', 'rdk5'))
    parser.add_argument('--bits', help="Number of fingerprint bits, default: %(default)s", default=1024, type=int)
    parser.add_argument('--schrodinger', help='Path to Schrodinger Suite root directory, default: %(default)s',
                        type=vstool.check_dir, default='/work/08944/fuzzy/share/software/DESRES/2023.2')
    parser.add_argument('--md', help='Path to md executable, default: %(default)s',
                        type=vstool.check_exe, default='/work/08944/fuzzy/share/software/virtual-screening/venv/lib/python3.11/site-packages/virtual_screening/desmond_md.sh')
    parser.add_argument('--time', type=float, default=50, help="MD simulation time, default: %(default)s ns.")

    parser.add_argument('--nodes', type=int, default=8, help="Number of nodes, default: %(default)s.")
    parser.add_argument('--email', help='Email address for send status change emails')
    parser.add_argument('--email-type', help='Email type for send status change emails, default: %(default)s',
                        default='ALL', choices=('NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'))
    parser.add_argument('--delay', help='Hours need to delay running the job.', type=int, default=0)
    
    parser.add_argument('--docking-only', help='Only perform docking and post-docking and no MD.', action='store_true')
    parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
    parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

    args = parser.parse_args()
    vstool.setup_logger(verbose=True)

    setattr(args, 'summary', args.outdir / 'docking.md.summary.csv')
    if args.summary.exists():
        vstool.debug_and_exit(f'Virtual screening result {args.summary} already exists, skip re-processing')
        
    setattr(args, 'pdb', args.pdb or vstool.check_file(str(args.receptor)[:-2]))
    setattr(args, 'pdb', vstool.check_file(args.pdb))
    setattr(args, 'scratch', vstool.mkdir(args.scratch / f'{args.outdir.name}'))
    setattr(args, 'nodes', 2 if args.debug else args.nodes)

    hostname = socket.gethostname()
    ntasks_per_node = 4 if 'frontera' in hostname else 3
    ntasks = args.nodes * ntasks_per_node
    gpu_queue = 'rtx' if 'frontera' in hostname else 'gpu-a100'

    env = (f'source {Path(vstool.check_exe("python")).parent}/activate\n\n'
           f'cd {args.outdir} || {{ echo "Failed to cd into {args.outdir}!"; exit 1; }}\n')

    (cx, cy, cz), (sx, sy, sz) = args.center, args.size
    cmd = ['batch-ligand', str(args.ligand), str(args.receptor),
           str(args.outdir), f'--outdir {args.scratch}',
           f'--batch {ntasks}', f'--center {cx} {cy} {cz}', f'--size {sx} {sy} {sz}',
           f'--pdb {args.pdb}', f'--top {args.top}', f'--clusters {args.clusters}',
           f'--method {args.method}', f'--bits {args.bits}',
           f'--schrodinger {args.schrodinger}', f'--summary {args.summary}']
    if args.filter:
        cmd.append(f'--filter {args.filter}')
    if args.flexible:
        cmd.append(f'--flexible {args.flexible}')
    if args.residue:
        cmd.append(f'--residue {" ".join(str(x) for x in residue)}')
    code, job_id = vstool.submit(cmding(env, cmd, args),
                                 nodes=1, job_name='batch.ligand', hour=1, minute=30,
                                 partition='flex' if 'frontera' in hostname else 'vm-small',
                                 email=args.email, mail_type=args.email_type,
                                 log='vs.log', mode='append', script=args.outdir / 'batch.ligand.sh', delay=args.delay)

    lcmd = ['module load launcher_gpu', 'export LAUNCHER_WORKDIR={outdir}',
            'export LAUNCHER_JOB_FILE={outdir}/{job}.commands.txt', '',
            '${{LAUNCHER_DIR}}/paramrun', '']

    code, job_id = vstool.submit('\n'.join(lcmd).format(outdir=str(args.outdir), job='docking'),
                                 nodes=args.nodes, ntasks=ntasks, ntasks_per_node=ntasks_per_node,
                                 job_name='docking', day=0 if args.debug else 1, hour=4 if args.debug else 12,
                                 partition=gpu_queue,
                                 email=args.email, mail_type=args.email_type,
                                 log='vs.log', mode='append', script=args.outdir / 'docking.sh',
                                 dependency=f'afterok:{job_id}', delay=args.delay)
    
    if args.docking_only:
        vs.debug_and_exit('Exit without submitting MD task due to docking only flag was set')
    
    outdir = vstool.mkdir(args.scratch / 'md')
    vstool.submit('\n'.join(lcmd).format(outdir=str(outdir), job='md'),
                  nodes=args.nodes, ntasks=ntasks, ntasks_per_node=ntasks_per_node,
                  job_name='md', day=0 if args.debug else 1, hour=8 if args.debug else 20,
                  partition=gpu_queue,
                  email=args.email, mail_type=args.email_type,
                  log='vs.log', mode='append', script=args.outdir / 'md.sh',
                  dependency=f'afterok:{job_id}', delay=args.delay)


if __name__ == '__main__':
    main()
