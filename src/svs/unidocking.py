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
parser.add_argument('ligand', help="Path to a SDF file contains prepared ligands")
parser.add_argument('receptor', help="Path to prepared rigid receptor in .pdbqt format file")
parser.add_argument('-f', '--flexible', help="Path to prepared flexible receptor in .pdbqt format file")
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
POSE, SCORE = OUTDIR / 'docking.pose.sdf.gz', OUTDIR / 'docking.scores.parquet'
if POSE.exists() and SCORE.exists():
    utility.debug_and_exit('Docking results already exist, skip re-docking', task=args.task, status=80)

utility.submit_or_skip(parser.prog, args,
                       ['ligand', 'receptor', 'field'],
                       ['flexible', 'filter', 'size', 'center', 'quiet', 'verbose', 'outdir', 'cpu',
                        'gpu', 'autodock', 'unidock', 'gnina', 'task', 'wd'])


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
utility.task_update(args.task, 70)

LIGANDS = list(LIGAND.glob('*.sdf')) + list(LIGAND.glob('*.sdf.gz'))
if args.debug:
    LIGANDS = LIGANDS[:4]


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

        ds = Descriptors.CalcMolDescriptors(m)
        num_chiral_center = len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=True))
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


def best_unidock_pose_and_score(sdf):
    ss = []
    for s in sdf_parser(sdf, deep=True):
        if s.properties_dict:
            try:
                score = s.properties_dict.get('<Uni-Dock RESULT>', '').splitlines()[0]
                score = float(score.split('=')[1].strip().split()[0])
            except ValueError as e:
                logger.error(f'Failed to get score from {sdf} due to {e}')
                score = np.nan
            ss.append((s.no_properties(), score))

    s, score = sorted(ss, key=lambda x: x[1])[0] if ss else ['', np.nan]
    return s, score


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


def unidock(batch, receptor=None, outdir=None, exe='', verbose=False):
    gpu_id, log = GPU_QUEUE.get(), Path(batch).with_suffix(".log").name
    (cx, cy, cz), (sx, sy, sz) = receptor['center'], receptor['size']
    cmd = (f'{exe} --receptor {receptor["rigid"]} '
           f'--ligand_index {batch} --devnum {gpu_id} --search_mode balance --scoring vina '
           f'--center_x {cx} --center_y {cy} --center_z {cz} --size_x {sx} --size_y {sy} --size_z {sz} '
           f'--dir {outdir.name} &> {log}')

    try:
        p = cmder.run(cmd, cwd=str(outdir.parent), log_cmd=verbose)
        if p.returncode:
            utility.error_and_exit(f'Docking ligands in {batch} failed', task=args.task, status=-80)
    finally:
        GPU_QUEUE.put(gpu_id)


@task(inputs=LIGANDS, outputs=lambda i: f"{i.name.removesuffix('.sdf.gz').removesuffix('.sdf')}.txt", cpus=CPUS)
def get_ligand_list(inputs, outputs):
    logger.debug(f'Processing ligands in {inputs}')
    sdf_outdir = SCRATCH / outputs.removesuffix('.txt')
    sdf_outdir.mkdir(exist_ok=True)

    if args.filter:
        with open(args.filter) as f:
            filters = json.load(f)
    else:
        filters = None

    opener = gzip.open if inputs.name.endswith('.gz') else open
    with opener(inputs) as f, open(outputs, 'w') as o:
        mols = (filtering(mol, filters=filters) for mol in Chem.ForwardSDMolSupplier(f, removeHs=False))
        if args.debug:
            for i, mol in enumerate(mols):
                o.write(f'{sdf_writer(mol, sdf_outdir)}\n')
                if i == 100:
                    break
        else:
            o.writelines(f'{sdf_writer(mol, sdf_outdir)}\n' for mol in mols)
    logger.debug(f'Successfully processed ligands in {inputs}')


@task(inputs=get_ligand_list, outputs=lambda i: f"{i.removesuffix('.txt')}.score.parquet", cpus=GPU_QUEUE.qsize())
def docking(inputs, outputs):
    receptor = get_receptor(FIELD, RECEPTOR, flexible=args.flexible, size=args.size, center=args.center)
    outdir = Path(inputs.removesuffix('.txt'))
    outdir.mkdir(exist_ok=True)

    logger.debug(f'Docking ligand in {inputs} using Uni-Dock ...')
    start = time.time()
    unidock(inputs, receptor=receptor, outdir=outdir.resolve(), exe=args.unidock, verbose=args.verbose)
    t = str(timedelta(seconds=time.time() - start))
    logger.debug(f'Docking ligand in {inputs} using Uni-Dock complete in {t.split(".")[0]}\n')

    logger.debug(f'Parsing docking scores in {outdir}')
    start = time.time()
    sdfs, scores = [], []

    for sdf in outdir.glob('*_out.sdf'):
        s, score = best_unidock_pose_and_score(sdf)
        sdfs.append(s)
        scores.append({'ligand': sdf.name.removesuffix('_out.sdf'), 'score': score})

    df = pd.DataFrame(scores)
    df = df.dropna()

    if df.empty:
        logger.warning('Docking score dataframe is empty, no score will be saved. Check docking log to see if '
                       'docking performed successfully')
    else:
        with open(f'{OUTDIR / outputs.removesuffix(".score.parquet")}.pose.sdf', 'w') as o:
            o.writelines(f'{sdf}\n' for sdf in sdfs if sdf)

        df.to_parquet(outputs, index=False)
        logger.debug(f'Successfully saved {df.shape[0]:,} docking scores to {outputs}')
    t = str(timedelta(seconds=time.time() - start))
    logger.debug(f'Parsing docking scores complete in {t.split(".")[0]}\n')


@task(inputs=[], outputs=[SCORE], parent=docking)
def scoring(inputs, outputs):
    cmder.run(f'cat *.log > {OUTDIR}/docking.log && rm *.log')
    cmder.run(f'rm *.txt')
    df = [pd.read_parquet(x) for x in Path('.').glob('*.score.parquet')]
    df = pd.concat(df)
    df.to_parquet(SCORE, index=False)
    cmder.run(f'rm *.score.parquet')
    cmder.run(f'rm -r {SCRATCH}', cwd=str(OUTDIR))


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
