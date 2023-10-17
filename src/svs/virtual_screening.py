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


METHODS = ('morgan2', 'morgan3', 'ap', 'rdk5')

parser = argparse.ArgumentParser(prog='virtual-screening', description=__doc__.strip())
parser.add_argument('ligand', help="Path to a single SDF file contains prepared ligands")
parser.add_argument('receptor', help="Path to prepared rigid receptor in .pdbqt format file")
parser.add_argument('field', help="Path to prepared receptor maps filed file in .maps.fld format file")

parser.add_argument('-p', '--pdb', help="Path to receptor structure in .pdbqt format file")
parser.add_argument('-f', '--flexible', help="Path to prepared flexible receptor in .pdbqt format file")
parser.add_argument('--filter', help="Path to a JSON file contains ligand filters")
parser.add_argument('-s', '--size', help="The size in the X, Y, and Z dimension (Angstroms)", type=int, nargs='+')
parser.add_argument('-c', '--center', help="T X, Y, and Z coordinates of the center", type=float, nargs='+')

parser.add_argument('--cpu', type=int, default=32,
                    help="Maximum number of CPUs can be used for parallel processing, default: %(default)s")
parser.add_argument('--gpu', type=int, default=4,
                    help="Maximum number of GPUs can be used for docking, default: %(default)s")

parser.add_argument('-o', '--outdir', default='.', type=str,
                    help="Path to a directory for saving output files, default: %(default)s")
parser.add_argument('--scratch', default='/scratch', type=str,
                    help="Path to the scratch directory, default: %(default)s")

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
                                                "default: %(default)s", type=float, default=50)
parser.add_argument('-r', '--residue', nargs='?', type=int,
                        help="Residue numbers that interact with ligand via hydrogen bond")
parser.add_argument('--schrodinger', help='Path to Schrodinger Suite root directory')
parser.add_argument('--openmm_simulate', help='Path to openmm_simulate executable')

parser.add_argument('-q', '--quiet', help='Process data quietly without debug and info message',
                    action='store_true')
parser.add_argument('-v', '--verbose', help='Process data verbosely with debug and info message',
                    action='store_true')

parser.add_argument('--wd', help="Path to work directory", default='.')
parser.add_argument('--task', type=int, help="An ID associated with the task", default=0)

parser.add_argument('--submit', action='store_true', help="Submit job to job queue instead of directly running it")
parser.add_argument('--hold', action='store_true', help="Hold the submission without actually submit the job to the queue")
parser.add_argument('--dry', help='Only print out tasks and commands without actually running them.',
                    action='store_true')

args = parser.parse_args()

utility.setup_logger(quiet=args.quiet, verbose=args.verbose)
log = f'{parser.prog}.log'

utility.submit_or_skip(parser.prog, args,
                       ['ligand', 'receptor', 'field'],
                       ['pdb', 'flexible', 'filter', 'size', 'center', 'cpu', 'gpu',
                        'outdir', 'scratch', 'autodock', 'unidock', 'gnina',
                        'top_percent', 'num_clusters', 'method', 'bits', 'md_time', 'residue',
                        'schrodinger', 'openmm_simulate',
                        'quiet', 'verbose', 'wd', 'task'],
                       day=21, log=log)

outdir = utility.make_directory(args.outdir, task=args.task, status=-1)
os.chdir(outdir)

ligand = utility.check_file(args.ligand, f'The provide ligand {args.ligand} does not exist or not a file',
                            task=args.task, status=-1)
receptor = utility.check_file(args.receptor, 
                              f'The provide receptor {args.receptor} does not exist or not a file', 
                              task=args.task, status=-1)
field = utility.check_file(args.field, f'The provide field {args.field} does not exist or not a file', 
                           task=args.task, status=-1)
activate = utility.check_executable(parser.prog, task=args.task, status=-2).strip().removesuffix('virtual-screening')
venv = f'source {activate}activate\ncd {outdir} || exit 1'


@task(inputs=[str(ligand)], outputs=['docking.scores.parquet'])
def molecule_docking(inputs, outputs):
    (sx, sy, sz), (cx, cy, cz) = args.size, args.center
    cmd = (f'unidock {ligand} {receptor} '
           f'--size {sx} {sy} {sz} --center {cx} {cy} {cz} '
           f'--outdir {outdir} --scratch {args.scratch} '
           f'--cpu {args.cpu} --gpu {args.gpu} '
           f'--exe {args.unidock} --task {args.task} --verbose')
    if args.filter:
        cmd += f' --filter {args.filter}'
    if args.flexible:
        cmd += f' --flexible {args.flexible}'

    p = cmder.run(cmd, debug=args.verbose)
    if p.returncode:
        utility.error_and_exit('Failed to run docking', task=args.task, status=-80)


@task(inputs=molecule_docking, outputs=['top.pose.sdf'])
def top_pose(inputs, outputs):
    cmd = f'top-pose {inputs} --top {args.top_percent} --task {args.task}'
    p = cmder.run(cmd, debug=args.verbose)
    if p.returncode:
        utility.error_and_exit('Get top poses failed', task=args.task, status=-95)


@task(inputs=top_pose, outputs=['cluster.pose.sdf'])
def cluster_pose(inputs, outputs):
    cmd = f'cluster-pose {inputs} --clusters {args.num_clusters} --cpu {args.cpu} --task {args.task}'
    p = cmder.run(cmd, debug=args.verbose)
    if p.returncode:
        utility.error_and_exit('Clustering poses failed', task=args.task, status=-105)


@task(inputs=cluster_pose, outputs=['interaction.pose.sdf'])
def interaction_pose(inputs, outputs):
    cmd = f'interaction-pose {inputs} {args.pdb} --schrodinger {args.schrodinger} --task {args.task}'
    if args.residue:
        cmd += f' --residue_number {" ".join(x for x in args.residume)}'
    p = cmder.run(cmd, debug=args.verbose)
    if p.returncode:
        utility.error_and_exit('Filtering interact poses failed', task=args.task, status=-115)


@task(inputs=interaction_pose, outputs=['md/RMSD.csv'])
def molecule_dynamics(inputs, outputs):
    cmd = (f'molecule-dynamics {inputs} {args.pdb} --outdir {outdir / "md"} --openmm_simulate {args.openmm_simulate} '
           f' --cpu {args.gpu*2} --gpu {args.gpu} --task {args.task}')
    if args.residue:
        cmd += f' --time {args.md_time}'
    else:
        cmd += f' --short 5 --time {args.md_time}'
    p = cmder.run(cmd, debug=args.verbose)
    if p.returncode:
        if args.residue:
            utility.error_and_exit('Long time MD failed', task=args.task, status=-135)
        else:
            utility.error_and_exit('Short time MD failed', task=args.task, status=-125)


def main():
    try:
        start = time.time()
        flow = Flow('virtual-screening', short_description=__doc__.splitlines()[0], description=__doc__)
        flow.run(dry_run=args.dry, cpus=args.cpu)
        t = str(timedelta(seconds=time.time() - start))
        utility.debug_and_exit(f'Virtual screening complete in {t.split(".")[0]}\n',
                               task=args.task, status=140)
    except Exception as e:
        utility.error_and_exit(f'Virtual screening failed due to\n{e}\n\n{traceback.format_exc()}\n',
                               task=args.task, status=-20)


if __name__ == '__main__':
    main()
