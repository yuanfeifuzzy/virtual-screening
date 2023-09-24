#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run molecule dynamics in an easy way
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import timedelta

import cmder
import utility
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from sklearn.cluster import MiniBatchKMeans

parser = argparse.ArgumentParser(prog='molecule-dynamics', description=__doc__.strip())
parser.add_argument('sdf', help="Path to a SDF file contains best docking pose for each cluster")
parser.add_argument('pdb', help="Path to a PDF file contains the structure for the docking target")
parser.add_argument('-o', '--outdir', default='.', type=str,
                    help="Path to a directory for saving output files, default: %(default)s")
parser.add_argument('-t', '--time', type=int, default=5, help="MD simulation time, default: %(default)s ns.")
parser.add_argument('-g', "--gpu", type=int, default=4, help="Number of GPUs can be used, default: %(default)s.")
parser.add_argument('-q', '--quiet', help='Process data quietly without debug and info message', action='store_true')

parser.add_argument('--openmm_simulate', help='Path to openmm_simulate executable')
parser.add_argument('--wd', help="Path to work directory", default='.')

parser.add_argument('--task', type=int, default=0, help="ID associated with this task")
parser.add_argument('--success', type=int, default=0, help="Success code for the task")
parser.add_argument('--submit', action='store_true', help="Submit job to job queue instead of directly running it")
parser.add_argument('--hold', action='store_true',
                    help="Hold the submission without actually submit the job to the queue")
parser.add_argument('--version', version=__VERSION__, action='version')

args = parser.parse_args()

GPU_QUEUE = utility.gpu_queue(n=args.gpu)


def openmm_md(item):
    gpu_id = GPU_QUEUE.get()
    mol, target, name, t, outdir = item
    sdf = f'{name}.sdf'
    Chem.SDWriter(sdf).write(mol)
    cmd = f'{args.openmm_simulate} {sdf} {target} {name} --time {t} > {name}.log'
    try:
        cmder.run(cmd, env={'CUDA_VISIBLE_DEVICES': str(gpu_id)}, cwd=outdir)
    finally:
        GPU_QUEUE.put(gpu_id)
    return output


def main():
    utility.setup_logger(quiet=args.quiet, verbose=args.verbose)
    status = 120 if args.time <= 5 else 130
    
    if args.submit or args.hold:
        prog, wd = parser.prog, utility.make_directory(args.wd, task=args.task, status=-1)
        activation = utility.check_executable(prog).strip().replace(prog, 'activation', task=args.task, status=-2)
        venv = f'source {activation}\ncd {wd}'
        cmdline = utility.format_cmd(prog, args, ['sdf', 'pdb'],
                                     ['output', 'time', 'gpu', 'openmm_simulate', 'quiet', 'verbose', 'task'])
        return_code, job_id = utility.submit(cmdline, venv=venv, name=prog, day=14, cpu=args.gpu, gpu=args.gpu,
                                             hold=args.hold, script=f'{prog}.sh')
        utility.update_status(return_code, job_id, args.task, status)
        sys.exit(0)
    else:
        sdf = utility.chech_file(args.sdf, task=args.task, status=-1)
        pdb = utility.chech_file(args.pdb, task=args.task, status=-1)
        outdir = utility.make_directory(args.outdir, task=args.task, status=-1)
        utility.check_executable(args.openmm_simulate, task=args.task, status=-2)
        
        try:
            start = time.time()
            df = PandasTools.LoadSDF(sdf, smilesName='smiles', molColName='Molecule', removeHs=False)
            if df.empty:
                utility.error_and_exit(f'No pose found in {sdf}, cannot continue', task=args.task, status=-1*status)
            
            poses = ((row.Molecule, str(pdb), row.Molecule.GetProp('_Name') or str(i), args.time, str(outdir))
                     for i, row in enumerate(df.itertuples()))
            utility.parallel_gpu_task(openmm_md, poses, queue=GPU_QUEUE)
            t = str(timedelta(seconds=time.time() - start))
            utility.debug_and_exit(f'MD complete in {t.split(".")[0]}\n', task=args.task, status=status)
        except Exception as e:
            utility.error_and_exit(f'MD failed due to {e}\n', task=args.task, status=-1*status)


if __name__ == '__main__':
    main()
