#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A command line tool for easy ligand-protein docking with variety docking software
"""
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
from loguru import logger
from seqflow import task, Flow
from pandarallel import pandarallel


parser = argparse.ArgumentParser(prog='ligand-docking', description=__doc__.strip())
parser.add_argument('ligand', help="Path to a directory contains prepared ligands or a parquet file "
                                   "contains path and descriptors to ligands")
parser.add_argument('receptor', help="Path to prepared rigid receptor in .pdbqt format file")
parser.add_argument('field', help="Path to prepared receptor maps filed file in .maps.fld format file")
parser.add_argument('-f', '--flexible', help="Path to prepared flexible receptor in .pdbqt format file")
parser.add_argument('-y', '--filter', help="Path to a JSON file contains descriptor filters")
parser.add_argument('-s', '--size', help="The size in the X, Y, and Z dimension (Angstroms)", type=float, nargs='+')
parser.add_argument('-c', '--center', help="T X, Y, and Z coordinates of the center", type=float, nargs='+')
parser.add_argument('-b', '--batch_size', default=1000, type=int, help='Maximum number of ligands on each mini batch')

parser.add_argument('-o', '--outdir', default='.', type=str,
                    help="Path to a directory for saving docking output files, default: %(default)s")

parser.add_argument('--cpu', type=int, default=4,
                    help="Maximum number of CPUs can be used for parallel processing, default: %(default)s")
parser.add_argument('--gpu', type=int, default=4,
                    help="Maximum number of GPUs can be used for parallel processing, default: %(default)s")

parser.add_argument('-a', '--autodock', help="Path to AutoDock-GPU executable")
parser.add_argument('-u', '--unidock', help="Path to Uni-Dock executable")
parser.add_argument('-g', '--gnina', help="Path to gnina executable")

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
parser.add_argument('--dry', help='Only print out tasks and commands without actually running them.',
                    action='store_true')

args = parser.parse_args()

utility.submit_or_skip(parser.prog, args,
                       ['ligand', 'receptor', 'field'],
                       ['flexible', 'filter', 'size', 'center', 'batch_size', 'quiet', 'verbose', 'outdir', 'cpu',
                        'gpu', 'autodock', 'unidock', 'gnina', 'task', 'wd'])

utility.setup_logger(quiet=args.quiet, verbose=args.verbose)
OUTDIR = utility.make_directory(args.outdir)
os.chdir(OUTDIR)

SCORE = 'docking.scores.parquet'
if Path(SCORE).exists():
    utility.debug_and_exit('Docking results already exist, skip re-docking\n', task=args.task, status=80)

LIGAND = utility.check_exist(args.ligand, f'Ligand {args.ligand} does not exist', task=args.task, status=-1)
RECEPTOR = utility.check_file(args.receptor, f'Rigid receptor {args.receptor} is not a file or does not exist',
                              task=args.task, status=-1)
FIELD = utility.check_file(args.field, f'Map filed {args.field} is not a file or does not exist',
                           task=args.task, status=-1)

if args.flexible:
    utility.check_file(args.flexible, f'Flexible receptor {args.flexible} is not a file or does not exist',
                       task=args.task, status=-1)
if args.filter:
    utility.check_file(args.filter, f'Filter JSON {args.filter} is not a file or does not exist',
                       task=args.task, status=-1)

utility.check_size_center(args.size, 'size', task=args.task, status=-3)
utility.check_size_center(args.center, 'center', task=args.task, status=-3)

if not args.dry:
    if args.autodock:
        utility.check_executable(args.autodock, task=args.task, status=-2)
    elif args.unidock:
        utility.check_executable(args.unidock, task=args.task, status=-2)
    elif args.gnina:
        utility.check_executable(args.gnina, task=args.task, status=-2)
    else:
        utility.error_and_exit('No docking software executable was provide, cannot continue', task=args.task,
                                 status=-2)

CPUS = utility.get_available_cpus(cpus=args.cpu)
GPU_QUEUE = utility.gpu_queue(n=args.gpu, task=args.task, status=-4)
pandarallel.initialize(nb_workers=CPUS, progress_bar=False, verbose=0)

job_id = os.environ.get('SLURM_JOB_ID', 0)
utility.task_update(args.task, 70)

LIGAND_LIST = ''
if LIGAND.is_file():
    if LIGAND.suffix == '.parquet':
        LIGAND_LIST = LIGAND
    else:
        utility.error_and_exit(f'Invalid ligand file {LIGAND}, only accepts a file contains ligand descriptors '
                                 f'in parquet format', task=args.task, status=-1)
elif LIGAND.is_dir():
    LIGAND_LIST = LIGAND / f'descriptor.parquet'
    if LIGAND_LIST.is_file():
        logger.debug(f'Found ligand descriptor file {LIGAND_LIST}')
    else:
        utility.error_and_exit(f'No ligand descriptor file was found, cannot continue', task=args.task, status=-1)
else:
    utility.error_and_exit(f'Invalid ligand {LIGAND}, only accept a descriptor file in parquet format '
                             f'or a directory contains prepared ligands and corresponding descriptor file',
                             task=args.task, status=-1)


def filtering(row, filters=None):
    mw = '' if filters['min_mw'] < row.MW <= filters['max_mw'] else f'MW={row.MW}'
    logP = '' if filters['min_logP'] < row.logP <= filters['max_logP'] else f'logP={row.logP}'
    hba = '' if row.HBA <= filters['hba'] else f'HBA={row.HBA}>{filters["hba"]}'
    hbd = '' if row.HBD <= filters['hbd'] else f'HBD={row.HBD}>{filters["hbd"]}'
    lipinski_violation = sum(x != '' for x in [mw, logP, hba, hbd])

    d = [
        mw,
        logP,
        '' if row.chiral_center <= filters['chiral_center']
        else f'chiral_center={row.chiral_center}>{filters["chiral_center"]}',
        '' if row.rotatable_bound_number <= filters['rotatable_bound_number']
        else f'rotatable_bound_number={row.rotatable_bound_number}>{filters["rotatable_bound_number"]}',
        hba,
        hbd,
        '' if row.TPSA <= filters['tpsa'] else f'TPSA={row.TPSA}>{filters["tpsa"]}',
        '' if row.QED >= filters['qed'] else f'QED={row.QED}<{filters["qed"]}',
        '' if lipinski_violation <= 1 else f'lipinski_violation={lipinski_violation}',
    ]
    return ';'.join([x for x in d if x])


@task(inputs=[LIGAND_LIST], outputs=['ligands.tsv'])
def get_ligand_list(inputs, outputs):
    logger.debug(f'Loading ligands and corresponding descriptors from {inputs}')
    df = utility.read(LIGAND_LIST)
    logger.debug(f'Successfully loaded {df.shape[0]:,} ligands and corresponding descriptors')

    if args.filter:
        with open(utility.check_file(args.filter)) as f:
            filters = json.load(f)

        logger.debug(f'Filtering {df.shape[0]:,} ligands with {len(filters)} descriptor filters')
        df['filter'] = df.parallel_apply(filtering, axis=1, filters=filters)
        df = df[df['filter'] == '']
        logger.debug(f'Successfully filtered ligands and there are {df.shape[0]:,} ligands passed all filters')

    if args.autodock:
        if 'pdbqt' in df:
            utility.write(df, outputs, columns=['pdbqt'], index=False, header=False)
        else:
            utility.error_and_exit(f'No prepared ligands in PDBQT format was found, cannot continue with AutoDock',
                                   task=args.task, status=-1)
    else:
        utility.write(df, outputs, columns=['sdf'], index=False, header=False)

    logger.debug(f'Successfully write {df.shape[0]:,} ligands to {outputs}')


def prepare_autodock_ligand_list(batch, field=''):
    output = batch.replace('ligands.', 'batches.')
    with open(batch) as f, open(output, 'w') as o:
        for line in f:
            out = Path(line.strip()).with_suffix('').name
            o.write(f'{line}{field}\n{out}\n')
    cmder.run(f'rm -f {batch}')
    return output


@task(inputs=get_ligand_list, outputs=['batches.tsv'])
def prepare_ligand_list(inputs, outputs):
    if args.gnina:
        cmder.run(f'mv {inputs} {outputs}')
    else:
        prefix = 'ligands.' if args.autodock else 'batches.'
        split = 'split --numeric-suffixes=1 --suffix-length=6'
        cmd = f'{split} --additional-suffix=.tsv --lines={args.batch_size} {inputs} {prefix}'
        cmder.run(cmd, exit_on_error=True)

        batches = list(OUTDIR.glob(f'{prefix}*.tsv'))
        if args.autodock:
            batches = utility.parallel_cpu_task(prepare_autodock_ligand_list, batches, processes=CPUS, chunksize=100,
                                                field=str(FIELD))
        with open(outputs, 'w') as o:
            o.write('\n'.join(str(batch) for batch in batches))

        cmder.run(f'rm -f {inputs}')


def get_size_center(fld):
    size, center = (0, 0, 0), (0, 0, 0)
    with open(fld) as f:
        for line in f:
            if line.startswith('#NELEMENTS'):
                try:
                    size = [int(x) for x in line.strip().split()[1:]]
                except ValueError:
                    print(f'Failed to get size coordinates from {fld}, size was set to {size}')
            if line.startswith('#CENTER'):
                try:
                    center = [float(x) for x in line.strip().split()[1:]]
                except ValueError:
                    print(f'Failed to get center coordinates from {fld}, center was set to {center}')
    return size, center


def get_receptor(field, rigid, flexible='', size=None, center=None):
    if not size or not center:
        size, center = get_size_center(field)

    receptor = {'field': field, 'rigid': rigid, 'flexible': flexible, 'size': size, 'center': center}
    return receptor


def autodock(batch, receptor=None, exe='', outdir='', verbose=False):
    gpu_id, log = GPU_QUEUE.get(), Path(batch).with_suffix(".log").name
    cmd = f'{exe} --filelist {batch} --devnum {int(gpu_id)+1} --xmloutput 0 --nrun 10 &> {log}'
    try:
        p = cmder.run(cmd, cwd=str(outdir), log_cmd=verbose, exit_on_error=True)
        cmder.run(f'rm -f {batch}')
    finally:
        GPU_QUEUE.put(gpu_id)
    return p.returncode


def unidock(batch, receptor=None, outdir=None, exe='', verbose=False):
    gpu_id, log = GPU_QUEUE.get(), Path(batch).with_suffix(".log").name
    (cx, cy, cz), (sx, sy, sz) = receptor['center'], receptor['size']
    cmd = (f'{exe} --receptor {receptor["rigid"]} '
           f'--ligand_index {batch} --devnum {gpu_id} --search_mode balance --scoring vina '
           f'--center_x {cx} --center_y {cy} --center_z {cz} --size_x {sx} --size_y {sy} --size_z {sz} '
           f'--dir {outdir} &> {log}')

    try:
        p = cmder.run(cmd, cwd=str(outdir), log_cmd=verbose, exit_on_error=True)
        cmder.run(f'rm -f {batch}')
    finally:
        GPU_QUEUE.put(gpu_id)
    return p.returncode


def gnina(ligand, receptor=None, outdir=None, exe='',  verbose=False):
    gpu_id , log = GPU_QUEUE.get(), Path(ligand).with_suffix(".log").name
    (cx, cy, cz), (sx, sy, sz) = receptor['center'], receptor['size']

    cmd = (f'{exe} --receptor {receptor["rigid"]} --ligand {ligand} --device {gpu_id} '
           f'--center_x {cx} --center_y {cy} --center_z {cz} --size_x {sx} --size_y {sy} --size_z {sz} '
           f'--out {outdir / Path(ligand).name} > {log}')

    try:
        p = cmder.run(cmd, cwd=str(outdir), log_cmd=verbose, exit_on_error=True)
    finally:
        GPU_QUEUE.put(gpu_id)
    return p.returncode


def get_score_from_autodock_dlg(dlg):
    scores = []
    with open(dlg) as f:
        for line in f:
            if 'Estimated Free Energy of Binding' in line:
                scores.append(float(line.split()[-3].strip('=')))
    return {'ligand': str(dlg), 'score': np.min(scores), 'idx': np.argmin(scores)}


def get_score_from_unidock_sdf(sdf):
    scores = []
    with open(sdf) as f:
        for line in f:
            if line.startswith('ENERGY='):
                scores.append(float(line.split()[1]))
    return {'ligand': str(sdf), 'score': np.min(scores), 'idx': np.argmin(scores)}


def get_score_form_gnina_sdf(sdf):
    scores = []
    with open(sdf) as f:
        for line in f:
            if '<minimizedAffinity>' in line:
                break
        scores.append(float(next(f).strip()))
    return {'ligand': str(sdf), 'score': np.min(scores), 'idx': np.argmin(scores)}


@task(inputs=prepare_ligand_list, outputs=[SCORE])
def docking(inputs, outputs):
    receptor = get_receptor(FIELD, RECEPTOR, flexible=args.flexible, size=args.size, center=args.center)
    with open(inputs) as f:
        batches = (line for line in f)
        if args.autodock:
            utility.parallel_gpu_task(autodock, batches, exe=args.autodock, receptor=receptor,
                                      outdir=OUTDIR, verbose=args.verbose, chunksize=100)
            df = utility.parallel_cpu_task(get_score_from_autodock_dlg, OUTDIR.glob('*.dlg'), processes=CPUS)
        elif args.unidock:
            utility.parallel_gpu_task(unidock, batches, exe=args.unidock, receptor=receptor,
                                      outdir=OUTDIR, verbose=args.verbose, chunksize=100)
            df = utility.parallel_cpu_task(get_score_from_unidock_sdf, OUTDIR.glob('*.sdf'), processes=CPUS)
        else:
            utility.parallel_gpu_task(gnina, batches, exe=args.gnina, receptor=receptor,
                                      outdir=OUTDIR, verbose=args.verbose, chunksize=1000)
            df = utility.parallel_cpu_task(get_score_form_gnina_sdf, OUTDIR.glob('*.sdf'), processes=CPUS)

        cmder.run(f'zip docking.log.zip batches.*.log')
        cmder.run(f'rm batches.*.log')

        df = pd.DataFrame(df)
        if df.empty:
            logger.warning('Docking score dataframe is empty, no score will be saved. Check docking log to see if '
                           'docking performed successfully')
        else:
            df.to_parquet(outputs, index=False)
            logger.debug(f'Successfully saved {df.shape[0]:,} docking scores to {outputs}')

    cmder.run(f'rm -f {inputs}')


def main():
    try:
        start = time.time()
        flow = Flow('sd', short_description=__doc__.splitlines()[0], description=__doc__)
        flow.run(dry_run=args.dry, cpus=CPUS)
        t = str(timedelta(seconds=time.time() - start))
        utility.debug_and_exit(f'Docking complete in {t.split(".")[0]}\n', task=args.task, status=80)
    except Exception as e:
        utility.error_and_exit(f'Docking failed due to\n{e}\n\n{traceback.format_exc()}', task=args.task, status=-80)


if __name__ == '__main__':
    main()
