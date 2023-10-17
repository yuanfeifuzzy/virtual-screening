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


parser = argparse.ArgumentParser(prog='ligand-docking', description=__doc__.strip())
parser.add_argument('ligand', help="Path to a directory contains prepared ligands in SDF format")
parser.add_argument('receptor', help="Path to prepared rigid receptor in .pdbqt format file")
parser.add_argument('field', help="Path to prepared receptor maps filed file in .maps.fld format file")
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
parser.add_argument('--debug', help='Enable debug mode (only for development purpose).',
                    action='store_true')
parser.add_argument('--dry', help='Only print out tasks and commands without actually running them.',
                    action='store_true')

args = parser.parse_args()

utility.submit_or_skip(parser.prog, args,
                       ['ligand', 'receptor', 'field'],
                       ['flexible', 'filter', 'size', 'center', 'quiet', 'verbose', 'outdir', 'cpu',
                        'gpu', 'autodock', 'unidock', 'gnina', 'task', 'wd'])

utility.setup_logger(quiet=args.quiet, verbose=args.verbose)
OUTDIR = utility.make_directory(args.outdir)

SCORE = OUTDIR / 'docking.scores.parquet'
if SCORE.exists():
    df = pd.read_parquet(SCORE)
    utility.debug_and_exit('Docking results already exist, skip re-docking\n', task=args.task, status=80)
    
SCRATCH = utility.make_directory(Path(args.scratch) / OUTDIR.name)
os.chdir(SCRATCH)

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

LIGANDS = list(LIGAND.glob('*.sdf')) + list(LIGAND.glob('*.sdf.gz'))
if args.debug:
    LIGANDS = LIGANDS[:GPU_QUEUE.qsize()]


def filtering(mol, filters=None):
    if filters:
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
        
    return mol


def sdf_writer(mol, outdir):
    output = str(outdir / f"{mol.GetProp('_Name')}.sdf")
    with Chem.SDWriter(output) as o:
        o.write(mol)
    return output


class SDF:
    def __init__(self, ss, deep=False):
        self.s = ss
        self.title, self.mol, self.properties, self.properties_dict = self._parse(ss, deep=deep)

    @staticmethod
    def _parse(s, deep=False):
        if s:
            lines = s.splitlines(keepends=True)
            title = lines[0].rstrip()

            mol, i = [], 0
            for i, line in enumerate(lines[1:]):
                mol.append(line)
                if line.strip() == 'M  END':
                    break
            mol = ''.join(mol)

            properties = lines[i + 2:]

            properties_dict, idx = {}, []
            if deep:
                for i, p in enumerate(properties):
                    if p.startswith('>'):
                        properties_dict[p.split(maxsplit=1)[1].rstrip()] = ''
                        idx.append(i)
                idx.append(-1)

                for i, k in enumerate(properties_dict.keys()):
                    properties_dict[k] = ''.join(properties[idx[i] + 1:idx[i + 1]])

            properties = ''.join(properties[:-1])
        else:
            title, mol, properties, properties_dict = '', '', '', {}
        return title, mol, properties, properties_dict

    def no_properties(self, output=None):
        s = f'{self.title}\n{self.mol}\n$$$$\n'
        if output:
            with open(output, 'w') as o:
                o.write(s)
        return s

    def __str__(self):
        return self.s


def sdf_parser(sdf, string=False, deep=False):
    opener = gzip.open if str(sdf).endswith('.gz') else open
    with opener(sdf) as f:
        lines = []
        for line in f:
            lines.append(line)
            if line.strip() == '$$$$':
                yield ''.join(lines) if string else SDF(''.join(lines), deep=deep)
                lines = []
                continue
        yield ''.join(lines) if string else SDF(''.join(lines), deep=deep)


def best_pose_and_score(sdf):
    ss = []
    for s in sdf_parser(sdf, deep=True):
        if s.properties_dict:
            try:
                score = s.properties_dict.get('<Uni-Dock RESULT>', '').splitlines()[0]
                score = float(score.split('=')[1].strip().split()[0])
            except ValueError as e:
                logger.error(f'Failed to get score from {sdf} due to {e}')
                score = np.nan
            ss.append((s.no_properties(), s.title, score))

    s, title, score = sorted(ss, key=lambda x: x[1])[0] if ss else ['', np.nan]
    return s, title, score


def get_size_center(fld):
    size, center = (0, 0, 0), (0, 0, 0)
    with open(fld) as f:
        for line in f:
            if line.startswith('#NELEMENTS'):
                try:
                    size = [int(x) for x in line.strip().split()[1:]]
                except ValueError:
                    logger.error(f'Failed to get size coordinates from {fld}, size was set to {size}')
            if line.startswith('#CENTER'):
                try:
                    center = [float(x) for x in line.strip().split()[1:]]
                except ValueError:
                    logger.error(f'Failed to get center coordinates from {fld}, center was set to {center}')
    return size, center


def get_receptor(field, rigid, flexible='', size=None, center=None):
    if not size or not center:
        size, center = get_size_center(field)

    receptor = {'field': field, 'rigid': rigid, 'flexible': flexible, 'size': size, 'center': center}
    return receptor


def unidock(batch, receptor=None, exe='', verbose=False):
    logger.debug(f'Docking ligands in {batch} ...')
    gpu_id, log = GPU_QUEUE.get(), Path(batch).with_suffix(".log").name
    (cx, cy, cz), (sx, sy, sz) = receptor['center'], receptor['size']
    cmd = (f'{exe} --receptor {receptor["rigid"]} '
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
        
        
def sdf_to_ligand_list(sdf):
    name = sdf.name.removesuffix('.sdf.gz').removesuffix('.sdf')
    outdir, output = SCRATCH / name, SCRATCH / name / f'{name}.txt'
    outdir.mkdir(exist_ok=True)
    
    if args.filter:
        with open(args.filter) as f:
            filters = json.load(f)
    else:
        filters = None
    
    opener = gzip.open if sdf.name.endswith('.gz') else open
    with opener(sdf) as f, open(output, 'w') as o:
        try:
            mols = (filtering(mol, filters=filters) for mol in Chem.ForwardSDMolSupplier(f, removeHs=False))
            if args.debug:
                for i, mol in enumerate(mols):
                    if mol:
                        o.write(f'{sdf_writer(mol, outdir)}\n')
                        if i > 100:
                            break
            else:
                o.writelines(f'{sdf_writer(mol, outdir)}\n' for mol in mols if mol)
        except Exception as e:
            logger.debug(f'Exception encountered during processing {sdf}:\n{e}\n\n{traceback.format_exc()}')
    logger.debug(f'Successfully processed ligands in {sdf}')
    return output


def main():
    try:
        start = time.time()
        
        libraries = utility.parallel_cpu_task(sdf_to_ligand_list, LIGANDS)
        
        receptor = get_receptor(FIELD, RECEPTOR, flexible=args.flexible, size=args.size, center=args.center)
        utility.parallel_gpu_task(unidock, libraries, receptor=receptor, exe=args.unidock, verbose=args.verbose)
        
        logger.debug(f'Getting best poses and scores ...')
        ps = utility.parallel_cpu_task(best_pose_and_score, SCRATCH.glob('*/*_out.sdf'))
        logger.debug(f'Getting best poses and scores complete.')
        
        logger.debug(f'Sorting and saving best poses and scores into {OUTDIR} ...')
        ps = sorted(ps, key=lambda x: x[2])
        
        with OUTDIR.joinpath('docking.pose.sdf').open('w') as o:
            o.writelines(f'{s[0]}\n$$$$\n' for s in ps)

        df = pd.DataFrame((s[1:] for s in ps), columns=['ligand', 'score'])
        df.to_parquet(SCORE, index=False)
        logger.debug(f'Sorting and saving best poses and scores into {OUTDIR} complete.')
        
        cmder.run(f'rm -r {SCRATCH}', msg='Cleaning up ...')
        
        t = str(timedelta(seconds=time.time() - start))
        utility.debug_and_exit(f'Docking complete in {t.split(".")[0]}\n', task=args.task, status=80)
    except Exception as e:
        utility.error_and_exit(f'Docking failed due to\n{e}\n\n{traceback.format_exc()}', task=args.task, status=-80)


if __name__ == '__main__':
    main()
