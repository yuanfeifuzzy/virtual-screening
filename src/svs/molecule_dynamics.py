#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run molecule dynamics in an easy way
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import timedelta

import cmder
import utility

from svs import tools

logger = utility.setup_logger()


parser = argparse.ArgumentParser(prog='molecule-dynamics', description=__doc__.strip())
parser.add_argument('sdf', help="Path to a SDF file contains best docking pose for each cluster")
parser.add_argument('pdb', help="Path to a PDF file contains the structure for the docking target")
parser.add_argument('-o', '--outdir', default='.', type=str,
                    help="Path to a directory for saving output files, default: %(default)s")
parser.add_argument('-t', '--time', type=float, default=5, help="MD simulation time, default: %(default)s ns.")
parser.add_argument('-c', "--cpu", type=int, default=4, help="Number of CPUs can be used, default: %(default)s.")
parser.add_argument('-g', "--gpu", type=int, default=4, help="Number of GPUs can be used, default: %(default)s.")
parser.add_argument('-q', '--quiet', help='Process data quietly without debug and info message', action='store_true')
parser.add_argument('-v', '--verbose', help='Process data verbosely with debug and info message',
                        action='store_true')

parser.add_argument('--openmm_simulate', help='Path to openmm_simulate executable')

parser.add_argument('--wd', help="Path to work directory", default='.')
parser.add_argument('--task', type=int, default=0, help="ID associated with this task")
parser.add_argument('--submit', action='store_true', help="Submit job to job queue instead of directly running it")
parser.add_argument('--hold', action='store_true',
                    help="Hold the submission without actually submit the job to the queue")
parser.add_argument('--version', version=utility.get_version(__package__), action='version')

args = parser.parse_args()

utility.setup_logger(quiet=args.quiet, verbose=args.verbose)

tools.submit_or_skip(parser.prog, args, ['sdf', 'pdb'],
                     ['outdir', 'time', 'gpu', 'openmm_simulate', 'quiet', 'verbose', 'task'], day=21)


GPU_QUEUE = utility.gpu_queue(n=args.gpu)


def openmm_md(item):
    print(item)
    gpu_id = GPU_QUEUE.get()
    print(gpu_id)
    mol, target, name, tt = item
    ligand = f'{name}.sdf'
    Chem.SDWriter(ligand).write(mol)
    print(f'MD simulate using {ligand}')
    logger.debug(f'MD simulate using {ligand}')
    cmd = f'{args.openmm_simulate} {ligand} {target} {name} --time {tt} &> {name}.log'
    try:
        p = cmder.run(cmd, env={'CUDA_VISIBLE_DEVICES': str(gpu_id)})
        print(cmd)
    finally:
        GPU_QUEUE.put(gpu_id)
    return p.returncode


sdf = utility.check_file(args.sdf, task=args.task, status=-1)
pdb = utility.check_file(args.pdb, task=args.task, status=-1)
outdir = utility.make_directory(args.outdir, task=args.task, status=-1)
utility.check_executable(args.openmm_simulate, task=args.task, status=-2)

try:
    start = time.time()
    os.chdir(outdir)
    
    with Chem.SDMolSupplier(str(sdf)) as f:
        poses = [(mol, str(pdb), mol.GetProp('_Name') or str(i), args.time) for i, mol in
                 enumerate(f)]
    print(f'Number of poses: {len(poses)}')
    utility.parallel_gpu_task(openmm_md, poses)
    t = str(timedelta(seconds=time.time() - start))
    utility.debug_and_exit(f'MD complete in {t.split(".")[0]}\n', task=args.task, status=125)
except Exception as e:
    utility.error_and_exit(f'MD failed due to\n{e}\n\n{traceback.format_exc()}', task=args.task, status=-1*125)


if __name__ == '__main__':
    main()
