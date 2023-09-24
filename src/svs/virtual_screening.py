#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A pipeline for perform virtual screening in an easy an smart way
"""
import json
import os
import sys
import argparse
import itertools
from pathlib import Path
from multiprocessing import Pool

import cmder
import utility
import pandas as pd
from rdkit import Chem
from slugify import slugify
from rdkit.Chem import Descriptors
from seqflow import task, Flow, logger

DEPENDENCY = os.environ.get('SLURM_JOB_ID', 0)
METHODS = ('morgan2', 'morgan3', 'ap', 'rdk5')

parser = argparse.ArgumentParser(prog='svs', description=__doc__.strip())
parser.add_argument('ligand', help="Path to a single file contains raw ligand or a directory "
                                   "contains prepared ligands")
parser.add_argument('receptor', help="Path to prepared rigid receptor in .pdbqt format file")
parser.add_argument('field', help="Path to prepared receptor maps filed file in .maps.fld format file")

parser.add_argument('-b', '--batch_size', default=2000, type=int,
                    help='Maximum number of items on each mini preparation or docking batch')
parser.add_argument('-y', '--filter', help="Path to a JSON file contains ligand filters")
parser.add_argument('--ligprep', help="Path to the executable of ligprep")
parser.add_argument('--gypsum', help="Path to the executable of gypsum_dl package")

parser.add_argument('-p', '--pdb', help="Path to receptor structure in .pdbqt format file")
parser.add_argument('-f', '--flexible', help="Path to prepared flexible receptor in .pdbqt format file")
parser.add_argument('-s', '--size', help="The size in the X, Y, and Z dimension (Angstroms)", type=float, nargs='+')
parser.add_argument('-c', '--center', help="T X, Y, and Z coordinates of the center", type=float, nargs='+')
parser.add_argument('--autodock', help="Path to AutoDock-GPU executable")
parser.add_argument('--unidock', help="Path to AutoDock-GPU executable")
parser.add_argument('--gnina', help="Path to AutoDock-GPU executable")

parser.add_argument('-t', '--top_percent', help="Percent of top poses need to be retained for "
                                                "downstream analysis, default: %(default)s", type=float, default=10)
parser.add_argument('-g', '--num_clusters', help="Number of clusters for clustering top poses, "
                                                "default: %(default)s", type=int, default=1000)
parser.add_argument('--method', help="Method for generating fingerprints, default: %(default)s",
                    default=METHODS[-1], choices=METHODS)
parser.add_argument('--bits', help="Number of fingerprint bits, default: %(default)s", default=1024, type=int)
parser.add_argument('-m', '--md_time', help="Time (in nanosecond) needs for molecule dynamics simulation, "
                                                "default: %(default)s", type=int, default=50)
parser.add_argument('-r', '--residue_number', help="Time (in ns) needs for molecule dynamics simulation, "
                                                "default: %(default)s", type=int)

parser.add_argument('-o', '--outdir', help="Path to a directory for saving output files", default='.')

parser.add_argument('--cpu', type=int, default=32,
                    help="Maximum number of CPUs can be used for parallel processing, default: %(default)s")
parser.add_argument('--gpu', type=int, default=4, 
                    help="Maximum number of GPUs can be used for docking, default: %(default)s")

parser.add_argument('-q', '--quiet', help='Process data quietly without debug and info message',
                    action='store_true')
parser.add_argument('-v', '--verbose', help='Process data verbosely with debug and info message',
                    action='store_true')
parser.add_argument('--task', type=int, help="An ID associated with the task")
parser.add_argument('--submit', action='store_true', help="Submit job to job queue instead of directly running it")
parser.add_argument('--hold', action='store_true', help="Hold the submission without actually submit the job to the queue")
parser.add_argument('--dry', help='Only print out tasks and commands without actually running them.',
                    action='store_true')

args = parser.parse_args()

utility.setup_logger(quiet=args.quiet, verbose=args.verbose)
LOG = f'{parser.prog}.log'

outdir = utility.make_directory(args.outdir, task=args.task, status=-1)
os.chdir(outdir)

if args.submit or args.hold:
    prog, wd = parser.prog, outdir
    activation = utility.check_executable(prog).strip().replace(prog, 'activation', task=args.task, status=-2)
    venv = f'source {activation}\ncd {wd}'
    cmdline = utility.format_cmd(prog, args, ['ligand', 'receptor', 'field'],
                                 ['pdb', 'flexible', 'filter', 'ligprep', 'gypsum', 'size', 'center', 'batch_size',
                                  'quiet', 'verbose', 'outdir', 'cpu', 'gpu', 'autodock', 'unidock', 'gnina',
                                  'top_percent', 'num_clusters', 'method', 'bits', 'md_time', 'residue_number'
                                  'task'])
    # options = utility.cmd_options(vars(args),
    #                               excludes=['ligand', 'receptor', 'field', 'submit', 'size', 'center', 'dry', 'hold'],
    #                               nargs=['size', 'center'])
    # cmdline = f'svs \\\n  {ligand} \\\n  {receptor} \\\n  {field} \\\n  {options}'

    return_code, job_id = utility.submit(cmdline, venv=venv, cpu=args.cpu, gpu=args.gpu, log=LOG,
                                         name=prog, day=14, script=f'{prog}.sh', hold=args.hold)
    utility.update_status(return_code, job_id, args.task, 1)
    sys.exit(0)
    

ligand = utility.check_exist(args.ligand, f'The provide ligand {args.ligand} does not exist')
receptor = utility.check_file(args.receptor, f'The provide receptor {args.receptor} does not exist or not a file')
field = utility.check_file(args.field, f'The provide field {args.field} does not exist or not a file')


@task(inputs=[ligand], outputs=[ligand.parent / 'descriptor.parquet'])
def prepare_ligand(inputs, outputs):
    global DEPENDENCY
    dd = {k: v for k, v in vars(args).items() if k in ('gypsum', 'ligprep', 'pdbqt', 'cpu', 'quiet', 'verbose', 'task')}
    cmd = SEPARATOR.join(['slp', str(inputs), f'--outdir {outputs.parent}', f'--batch_size {args.ligand_batch_size}',
                          utility.cmd_options(dd, indent=INDENT)])

    _, DEPENDENCY = utility.run_or_submit(cmd, dependency=DEPENDENCY, venv=VENV, cpu=args.cpu, log=LOG,
                                          name='slp', day=10, hour=0, hold=args.hold)


@task(inputs=prepare_ligand, outputs=['docking/docking.score.parquet'], mkdir=['docking'])
def molecule_docking(inputs, outputs):
    global DEPENDENCY
    kws = ('pdb', 'flexible', 'filter', 'cpu', 'gpu', 'autodock', 'unidock', 'gnina', 'task', 'quiet', 'verbose')
    nargs = ('size', 'center')
    dd = {k: v for k, v in vars(args).items() if k in kws or k in nargs}
    cmd = SEPARATOR.join(['sd', str(ligand), str(args.receptor), str(args.field), f'--outdir {outdir / "docking"}',
                          f'--batch_size {args.docking_batch_size}', utility.cmd_options(dd, indent=INDENT)])
    
    _, DEPENDENCY = utility.run_or_submit(cmd, dependency=DEPENDENCY, venv=VENV, cpu=args.cpu, gpu=args.gpu,
                                          log=LOG, name='sd', day=10, hour=0, hold=args.hold)


@task(inputs=molecule_docking, outputs=['top.pose.sdf'])
def top_pose(inputs, outputs):
    global DEPENDENCY
    dd = {k: v for k, v in vars(args).items() if k in ('cpu', 'task', 'quiet', 'verbose')}
    cmd = SEPARATOR.join(['top-pose', str(outdir / inputs),  f'--top {args.top_percent}',
                          f'--output {outdir / outputs}', utility.cmd_options(dd, indent=INDENT)])
    
    _, DEPENDENCY = utility.run_or_submit(cmd, dependency=DEPENDENCY, venv=VENV, cpu=args.cpu, log=LOG,
                                          name='top-pose', day=0, hour=16, hold=args.hold)


@task(inputs=top_pose, outputs=['cluster.pose.sdf'])
def cluster_pose(inputs, outputs):
    global DEPENDENCY
    dd = {k: v for k, v in vars(args).items() if k in ('cpu', 'method', 'bits', 'quiet', 'verbose', 'task')}
    cmd = SEPARATOR.join(['cluster-pose', f'{outdir / inputs}', f'--output {outdir / outputs}',
                          f'clusters {args.num_clusters}', utility.cmd_options(dd, indent=INDENT)])
    
    _, DEPENDENCY = utility.run_or_submit(cmd, dependency=DEPENDENCY, venv=VENV, cpu=args.cpu, log=LOG,
                                          name='cluster_pose', day=0, hour=16, hold=args.hold)


@task(inputs=cluster_pose, outputs=['interaction.pose.sdf'])
def filter_interaction(inputs, outputs):
    pass


@task(inputs=filter_interaction, outputs=['mo.pose.sdf'])
def quick_md(inputs, outputs):
    pass


@task(inputs=quick_md, outputs=['svs.pose.sdf'])
def deep_ma(inputs, outputs):
    pass


def main():
    flow = Flow('svs', short_description=__doc__.splitlines()[0], description=__doc__)
    flow.run(dry_run=args.dry, cpus=args.cpu)


if __name__ == '__main__':
    main()
