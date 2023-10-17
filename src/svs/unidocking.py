#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A command line tool for easy ligand-protein docking with variety docking software
"""
import gzip
import json
import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import timedelta

import utility
import cmder
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from loguru import logger
from seqflow import task, Flow
from pandarallel import pandarallel

from svs import sdf_io

parser = argparse.ArgumentParser(prog='ligand-docking', description=__doc__.strip())
parser.add_argument('ligand', help="Path to a SDF file contains prepared ligands")
parser.add_argument('receptor', help="Path to prepared rigid receptor in .pdbqt format file")
parser.add_argument('-f', '--flexible', help="Path to prepared flexible receptor in .pdbqt format file")
parser.add_argument('-y', '--filter', help="Path to a JSON file contains descriptor filters")
parser.add_argument('-s', '--size', help="The size in the X, Y, and Z dimension (Angstroms)", type=int, nargs='+')
parser.add_argument('-c', '--center', help="T X, Y, and Z coordinates of the center", type=float, nargs='+')

parser.add_argument('-o', '--outdir', default='.', type=str,
                    help="Path to a directory for saving docking output files, default: %(default)s")
parser.add_argument('--scratch', default='/scratch', type=str,
                    help="Path to the scratch directory, default: %(default)s")

parser.add_argument('--cpu', type=int, default=4,
                    help="Maximum number of CPUs can be used for parallel processing, default: %(default)s")
parser.add_argument('--gpu', type=int, default=4,
                    help="Maximum number of GPUs can be used for parallel processing, default: %(default)s")

parser.add_argument('--exe', help="Path to Uni-Dock executable")

parser.add_argument('-q', '--quiet', help='Process data quietly without debug and info message',
                    action='store_true')
parser.add_argument('-v', '--verbose', help='Process data verbosely with debug and info message',
                    action='store_true')

parser.add_argument('--wd', help="Path to work directory", default='.')
parser.add_argument('--task', type=int, default=0, help="ID associated with this task")

parser.add_argument('--submit', action='store_true', help="Submit job to job queue instead of directly running it")
parser.add_argument('--hold', action='store_true',
                    help="Hold the submission without actually submit the job to the queue")
parser.add_argument('--version', version=utility.get_version(__package__), action='version')
parser.add_argument('--debug', help='Enable debug mode (only for development purpose).',
                    action='store_true')
parser.add_argument('--dry', help='Only print out tasks and commands without actually running them.',
                    action='store_true')

args = parser.parse_args()
utility.setup_logger(quiet=args.quiet, verbose=args.verbose)

OUTDIR = Path(args.outdir)
POSE, SCORE = OUTDIR / 'docking.poses.sdf.gz', OUTDIR / 'docking.scores.parquet'
if POSE.exists() and SCORE.exists():
    utility.debug_and_exit('Docking results already exist, skip re-docking', task=args.task, status=80)

utility.submit_or_skip(parser.prog, args,
                       ['ligand', 'receptor'],
                       ['flexible', 'filter', 'size', 'center', 'outdir', 'scratch', 'cpu', 'gpu', 'exe',
                        'quiet', 'verbose', 'task', 'wd'])


OUTDIR = utility.make_directory(args.outdir)
SCRATCH = utility.make_directory(Path(args.scratch) / OUTDIR.name)
os.chdir(SCRATCH)

LIGAND = utility.check_file(args.ligand, f'Ligand {args.ligand}  is not a file or does not exist', task=args.task, status=-1)
RECEPTOR = utility.check_file(args.receptor, f'Rigid receptor {args.receptor} is not a file or does not exist',
                              task=args.task, status=-1)

if args.flexible:
    utility.check_file(args.flexible, f'Flexible receptor {args.flexible} is not a file or does not exist',
                       task=args.task, status=-1)

utility.check_size_center(args.size, 'size', task=args.task, status=-3)
utility.check_size_center(args.center, 'center', task=args.task, status=-3)

if not args.dry:
    if args.exe:
        utility.check_executable(args.exe, task=args.task, status=-2)
    else:
        utility.error_and_exit('No docking software executable was provide, cannot continue', task=args.task,
                               status=-2)

CPUS = utility.get_available_cpus(cpus=args.cpu)
GPU_QUEUE = utility.gpu_queue(n=args.gpu, task=args.task, status=-4)
pandarallel.initialize(nb_workers=CPUS, progress_bar=False, verbose=0)

job_id = os.environ.get('SLURM_JOB_ID', 0)
utility.task_update(task=args.task, status=70)


def sdf_to_ligand_list(sdf):
    logger.debug(f'Processing ligands in {sdf} ...')
    name = sdf.name.removesuffix('.sdf.gz').removesuffix('.sdf')
    outdir = SCRATCH / name
    outdir.mkdir(exist_ok=True)

    ligands, outputs, n = [], [], 0
    number = 100 if args.debug else 10000
    for i, ligand in enumerate(sdf_io.parse(sdf), 1):
        output = f'{outdir}/{ligand.title}.sdf'
        ligand.sdf(output=output)
        ligands.append(output)
        if i % number == 0:
            n += 1

            output = SCRATCH / f'{name}.{n:06d}.txt'
            with output.open('w') as o:
                o.write('\n'.join(ligands))
            outputs.append(output)

            ligands = []
            if args.debug and n == 9:
                break

    logger.debug(f'Successfully processed ligands in {sdf} into {n:,} file lists')
    return outputs


def filtering(sdf, filters):
    try:
        mol = next(Chem.SDMolSupplier(sdf, removeHs=False))
    except Exception as e:
        logger.error(f'Failed to read {sdf} deu to \n{e}\n\n{traceback.format_exc()}')
        return

    mw = Descriptors.MolWt(mol)
    if mw == 0 or mw < filters['min_mw'] or mw >= filters['max_mw']:
        return

    hba = Descriptors.NOCount(mol)
    if hba > filters['hba']:
        return

    hbd = Descriptors.NHOHCount(mol)
    if hbd > filters['hbd']:
        return

    logP = Descriptors.MolLogP(mol)
    if logP < filters['min_logP'] or logP > filters['max_logP']:
        return

    # https://www.rdkit.org/docs/GettingStartedInPython.html#lipinski-rule-of-5
    lipinski_violation = sum([mw <= 500, hba <= 10, hbd <= 5, logP <= 5])
    if lipinski_violation < 3:
        return

    ds = Descriptors.CalcMolDescriptors(mol)
    num_chiral_center = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=True))
    if num_chiral_center > filters["chiral_center"]:
        return

    if ds.get('NumRotatableBonds', 0) > filters['rotatable_bound_number']:
        return

    if ds.get('TPSA', 0) > filters['tpsa']:
        return

    if ds.get('qed', 0) > filters['qed']:
        return

    return sdf


def filter_ligands(txt, filters=None):
    output = Path(txt).with_suffix('.filter.txt')
    if filters:
        if output.exists():
            logger.debug(f'Filtered ligands already exist in {output}, skip re-filtering')
        else:
            if isinstance(filters, (str, Path)):
                with open(filters) as f:
                    filters = json.load(f)

            logger.debug(f'Filtering ligands in {txt}')
            with open(txt) as f:
                outs = [filtering(line.strip(), filters) for line in f]
                outs = [out for out in outs if out]
                if outs:
                    with output.open('w') as o:
                        o.writelines(f'{out}\n' for out in outs)
                    logger.debug(f'Successfully saved {len(outs):,} ligands passed filters to {output}')
                else:
                    logger.debug(f'No ligands passed filters were saved to {output}')
                    return None
    else:
        with open(txt) as f, output.open('w') as o:
            o.writelines(line for line in f)
        logger.debug(f'No filters were provided, copy ligands in {txt} to {output}')
    return output


def unidock(batch, receptor=None, exe='', verbose=False):
    logger.debug(f'Docking ligands in {batch} ...')
    gpu_id, log = GPU_QUEUE.get(), Path(batch).with_suffix(".log").name
    (cx, cy, cz), (sx, sy, sz) = args.center, args.size
    cmd = (f'{exe} --receptor {receptor} '
           f'--ligand_index {batch} --devnum {gpu_id} --search_mode balance --scoring vina '
           f'--center_x {cx} --center_y {cy} --center_z {cz} --size_x {sx} --size_y {sy} --size_z {sz} '
           f'--dir {batch.parent} &> {log}')

    try:
        p = cmder.run(cmd, log_cmd=verbose)
        if p.returncode:
            utility.error_and_exit(f'Docking ligands in {batch} failed', task=args.task, status=-80)
        logger.debug(f'Docking ligands in {batch} complete.')
    finally:
        GPU_QUEUE.put(gpu_id)


def best_pose_and_score(sdf):
    ss = []
    for s in sdf_io.parse(sdf, deep=True):
        if s.properties_dict:
            try:
                score = s.properties_dict.get('<Uni-Dock RESULT>', '').splitlines()[0]
                score = float(score.split('=')[1].strip().split()[0])
                title = f'{s.title}_{score}'
            except ValueError as e:
                logger.error(f'Failed to get score from {sdf} due to {e}')
                score = np.nan
                title = ''
            ss.append((s.sdf(title=title), s.title, score))

    s, title, score = sorted(ss, key=lambda x: x[1])[0] if ss else ['', np.nan]
    return s, title, score


def docking():
    ligands = sdf_to_ligand_list(LIGAND)
    libraries = utility.parallel_cpu_task(filter_ligands, ligands, filters=args.filter)
    libraries = [ligand for ligand in libraries if ligand]

    utility.parallel_gpu_task(unidock, libraries, receptor=RECEPTOR, exe=args.exe, verbose=args.verbose)

    logger.debug(f'Getting best poses and scores ...')
    ps = utility.parallel_cpu_task(best_pose_and_score, SCRATCH.glob('*_out.sdf'))
    logger.debug(f'Getting best poses and scores complete.')

    logger.debug(f'Sorting and saving best poses and scores into {OUTDIR} ...')
    ps = sorted(ps, key=lambda x: x[2])

    with gzip.open(POSE, 'wt') as o:
        o.writelines(f'{s[0]}\n' for s in ps)

    df = pd.DataFrame((s[1:] for s in ps), columns=['ligand', 'score'])
    df.to_parquet(SCORE, index=False)
    logger.debug(f'Sorting and saving best poses and scores into {OUTDIR} complete.')

    cmder.run(f'rm -r {SCRATCH}', msg='Cleaning up ...')


def main():
    try:
        start = time.time()
        docking()
        t = str(timedelta(seconds=time.time() - start))
        utility.debug_and_exit(f'Docking complete in {t.split(".")[0]}\n', task=args.task, status=80)
    except Exception as e:
        utility.error_and_exit(f'Docking failed due to\n{e}\n\n{traceback.format_exc()}', task=args.task, status=-80)


if __name__ == '__main__':
    main()
