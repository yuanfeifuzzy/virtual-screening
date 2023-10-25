#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A pipeline for perform virtual screening in an easy an smart way
"""
import json
import os
import sys
import time
import argparse
import itertools
import traceback
from pathlib import Path
from datetime import timedelta
from multiprocessing import Pool

import cmder
import utility
import pandas as pd
from rdkit import Chem
from slugify import slugify
from rdkit.Chem import Descriptors
from seqflow import task, Flow, logger


parser = argparse.ArgumentParser(prog='virtual-screening', description=__doc__.strip())
parser.add_argument('ligand', help="Path to a directory contains prepared ligands in SDF format",
                    type=vstool.check_file)
parser.add_argument('receptor', help="Path to prepared rigid receptor in PDBQT format file", type=vstool.check_file)
parser.add_argument('--pdb', help="Path to prepared receptor structure in PDB format")
parser.add_argument('--flexible', help="Path to prepared flexible receptor file in PDBQT format")
parser.add_argument('--filter', help="Path to a JSON file contains descriptor filters")
parser.add_argument('--size', help="The size in the X, Y, and Z dimension (Angstroms)", type=int, nargs='+')
parser.add_argument('--center', help="T X, Y, and Z coordinates of the center", type=float, nargs='+')

parser.add_argument('--outdir', help="Path to a directory for saving output files", type=vstool.mkdir)
parser.add_argument('--docker', help="Path to docking program executable", type=vstool.check_exe)

parser.add_argument('--scratch', help="Path to the scratch directory, default: %(default)s",
                    default=Path(os.environ.get('SCRATCH', '/scratch')))
parser.add_argument('--cpu', type=int, default=0,
                    help="Maximum number of CPUs can be used for parallel processing, default: %(default)s")
parser.add_argument('--gpu', type=int, default=0,
                    help="Maximum number of GPUs can be used for parallel processing, default: %(default)s")

parser.add_argument('--batch', type=int, help="Number of batches that the docking job will be split to, "
                                              "default: %(default)s", default=8)
parser.add_argument('--task', type=int, default=0, help="ID associated with this task")

parser.add_argument('--residue', nargs='*', type=int,
                        help="Residue numbers that interact with ligand via hydrogen bond")
parser.add_argument('--top', help="Percentage of top poses need to be retained for "
                                                "downstream analysis, default: %(default)s", type=float, default=10)
parser.add_argument('--clusters', help="Number of clusters for clustering top poses, "
                                                "default: %(default)s", type=int, default=1000)
parser.add_argument('--method', help="Method for generating fingerprints, default: %(default)s",
                    default='morgan2', choices=('morgan2', 'morgan3', 'ap', 'rdk5'))
parser.add_argument('--bits', help="Number of fingerprint bits, default: %(default)s", default=1024, type=int)
parser.add_argument('--schrodinger', help='Path to Schrodinger Suite root directory', type=vstool.check_dir)

parser.add_argument('--time', type=float, default=50, help="MD simulation time, default: %(default)s ns.")
parser.add_argument('--openmm_simulate', help='Path to openmm_simulate executable')

parser.add_argument('--partition', help='Name of the queue, default: %(default)s', default='normal')
parser.add_argument('--email', help='Email address for send status change emails')
parser.add_argument('--email-type', help='Email type for send status change emails, default: %(default)s',
                    default='ALL', choices=('NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'))

parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
vstool.setup_logger(quiet=args.quiet, verbose=args.verbose)

setattr(args, 'result', args.outdir / 'md.rmsd.csv')
setattr(args, 'scratch', vstool.mkdir(args.scratch / f'{args.outdir.name}', task=args.task))
setattr(args, 'cpu', 4 if args.debug else args.cpu)
setattr(args, 'gpu', 2 if args.debug else args.gpu)
setattr(args, 'batch', 2 if args.debug else args.batch)

args = parser.parse_args()
utility.setup_logger(quiet=args.quiet, verbose=args.verbose)

if args.result.exists():
    vstool.debug_and_exit(f'Virtual screening result {args.result} already exists, skip re-processing', 
                          task=args.task, status=140)

env = (f'source {args.bin}/activate\n\n'
       f'cd {args.outdir} || {{ echo "Failed to cd into {args.outdir}!"; exit 1; }}\n')

cmd = ' \\\n  '.join(['batch-ligand', args.ligand, f'--outdir {args.outdir}', f'--batch {args.batch}',
                      f'--task {args.task}'])
code, job_id = vstool.submit(env + cmd,
                             cpus_per_task=1, job_name='batch.ligand', hour=0, minute=30,
                             partition=args.partition, email=args.email, mail_type=args.email_type,
                             log='%x.%j.log', script='batch.ligand.sh', hold=args.hold)
if not job_id:
    vstool.error_and_eixt('Failed to submit batch ligand job, cannot continue', ta=args.task, status=-5)

cmd = ['docking',
       str(args.outdir / 'batch."${{SLURM_ARRAY_TASK_ID}}".sdf'),
       str(args.receptor),
       f'--size {args.size[0]} {args.size[1]} {args.size[2]}',
       f'--center {args.center[0]} {args.center[1]} {args.center[2]}',
       f'--cpu {args.cpu}', f'--gpu {args.gpu}',
       f'--scratch {args.scratch}', f'--exe {args.dcoker.strip()}', f'--task {args.task}']
if args.flexible:
    cmd.append(f'--flexible {args.flexible}')

if args.filter:
    cmd.append(f'--filter {args.filter}')

if args.debug:
    cmd.append('--debug')

code, job_id = vstool.submit(env + ' \\\n  '.join(cmd),
                             cpus_per_task=args.cpu, gpus_per_task=args.gpu,
                             job_name='docking', day=1, hour=12, array=f'1-{args.batch}',
                             partition=args.partition, email=args.email, mail_type=args.email_type,
                             log='%x.%j.%A.%a.log', script='docking.sh', dependency=f'afterok:{job_id}')

if not job_id:
    vstool.error_and_eixt('Failed to submit docking job, cannot continue', ta=args.task, status=-5)

cmd = ['post-docking', str(args.sdf), str(args.pdb), f'--top {args.top}', f'--clusters {args.clusters} ',
       f'--method {args.method}', f'--bits {args.bits} ', f'--cpu {args.cpu}', f'--task {args.task}',
       f'--schrodinger {args.schrodinger}']
if args.residue:
    cmd.append(f'--residue {" ".join(str(x) for x in args.residue)}')

code, job_id = vstool.submit(env + ' \\\n  '.join(cmd), cpus_per_task=args.cpu, job_name='post.docking',
                             day=0, hour=12, partition=args.partition, email=args.email,
                             mail_type=args.email_type, log='%x.%j.log', script='post.docking.sh',
                             dependency=f'afterok:{job_id}')

if not job_id:
    vstool.error_and_eixt('Failed to submit post docking analysis job, cannot continue', ta=args.task,
                          status=-5)

cmd = ['molecule-dynamics', str(args.sdf), str(args.pdb), f'--outdir {args.outdir}', f'--time {args.time} ',
       f'--cpu {args.cpu}', f'--gpu {args.gpu}', f'--task {args.task}',
       f'--openmm_simulate {args.openmm_simulate}']

vstool.submit(env + ' \\\n  '.join(cmd),
              cpus_per_task=args.cpu,
              gpus_per_task=args.gpu, job_name='md', day=2, hour=0, array=f'1-{args.batch}',
              partition=args.partition, email=args.email, mail_type=args.email_type, log='%x.%j.%A.%a.log',
              script='md.sh', dependency=f'afterok:{job_id}')
