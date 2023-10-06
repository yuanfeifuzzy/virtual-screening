#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare ligands in variety file formats into 3-d coordinates in (sdf or pdbqt format), ready for molecular docking
"""

import os
import sys
import time
import argparse
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
from pandarallel import pandarallel
from seqflow import task, Flow, logger


parser = argparse.ArgumentParser(prog='ligand-preparation', description=__doc__.strip())
parser.add_argument('source', help="Path to a single file contains raw ligands need to be prepared or "
                                   "a directory contains prepared ligands")
parser.add_argument('-o', '--outdir', help="Path to a output directory for saving prepared ligands", default='.')
parser.add_argument('-l', '--ligprep', help="Path to the executable of ligprep")
parser.add_argument('-g', '--gypsum', help="Path to the executable of gypsum_dl package")
parser.add_argument('-p', '--pdbqt', help="Prepare ligands in pdbqt format", action='store_true')
parser.add_argument('-b', '--batch_size', default=1000, type=int, help='Maximum number of ligands on each mini batch')
parser.add_argument('-c', '--cpu', default=32, type=int,
                    help='Number of maximum processors (CPUs) can be use for processing data')
parser.add_argument('-q', '--quiet', help='Process data quietly without debug and info message', action='store_true')
parser.add_argument('-v', '--verbose', help='Process data verbosely with debug and info message', action='store_true')

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
                       ['source'],
                       ['outdir', 'ligprep', 'gypsum', 'pdbqt', 'batch_size', 'cpu', 'quiet', 'verbose',
                        'wd', 'task'])

utility.setup_logger(quiet=args.quiet, verbose=args.verbose)
outdir = utility.make_directory(args.outdir, task=args.task, status=-2)
os.chdir(outdir)

DESCRIPTORS = f'descriptor.parquet'
if Path(DESCRIPTORS).exists():
    utility.debug_and_exit('Docking results already exist, skip re-docking\n', task=args.task, status=40)

source = utility.check_exist(args.source, f'The source of raw ligand {args.source} does not exist', task=args.task,
                             status=-1)
if not args.dry:
    if args.ligprep:
        utility.check_executable(args.ligprep, task=args.task, status=-2)
    elif args.gypsum:
        utility.check_executable(args.gypsum, task=args.task, status=-2)
    else:
        utility.error_and_exit('No ligand prepare software executable was provide, cannot continue', 
                                 task=args.task, status=-2)

if source.is_file():
    logger.debug(f'Prepare ligands listed in single file {source}')
elif source.is_dir():
    logger.debug(f'Checking prepared ligands in directory {source}')
    descriptor = source.joinpath('ligand.descriptors.parquet').resolve()
    if descriptor.is_file():
        utility.debug_and_exit(f'Found ligand descriptors in {descriptor} skip re-prepare ligands', task=args.task,
                               status=40)
    else:
        utility.error_and_exit(f'A directory was given for prepared ligand, but no ligands.descriptors.parquet '
                                 f'file was found', task=args.task, status=-40)
else:
    utility.error_and_exit(f'{source} is not a file or directory, cannot continue', task=args.task, status=-2)

logger.debug(f'Output directory was set to {outdir}')

CPUS = utility.get_available_cpus(cpus=args.cpu)
pandarallel.initialize(nb_workers=CPUS, progress_bar=False, verbose=0)

utility.task_update(args.task, 30)


@task(inputs=[source], outputs=[f'ligand.smiles'])
def ligand_to_smiles(inputs, smiles):
    smi = f'{smiles[:-7]}.smi'
    cmder.run(f'obabel {inputs} -O {smi}', exit_on_error=True)
    df = pd.read_csv(smi, sep='\t', header=None)
    if df.shape[1] == 2:
        logger.debug('Found names for SMILES in ligand file')
    else:
        logger.debug('No name was found for SMILES in ligand file, adding index as their names')
        df['uid'] = df.index.tolist()
        
    df.to_csv(smiles, sep='\t', index=False, header=False)
    cmder.run(f'rm -f {smi}')


def read(path):
    with open(path) as f:
        return [line.strip() for line in f]


def write(path, lines):
    with open(path, 'w') as o:
        o.writelines(f'{line}\n' for line in lines)


@task(inputs=ligand_to_smiles, outputs=lambda i: Path(i).with_suffix('.batches.txt').name)
def batch_smiles(smiles, output):
    ligand_name = output.rsplit('.', maxsplit=2)[0]
    split = 'split --numeric-suffixes=1 --suffix-length=5'
    cmd = f'{split} --additional-suffix=.smi --lines={args.batch_size} {smiles} {ligand_name}.'
    cmder.run(cmd, exit_on_error=True)
    batches = list(Path('.').glob(f'{ligand_name}.*.smi'))
    cmder.run(f'rm -f {smiles}')
    write(output, batches)
    logger.debug(f'Successfully split SMILES into {len(batches):,} batches')


def smi_to_sdf(smi):
    basename = str(smi)[:-4]
    if args.ligprep:
        sdf = f'{basename}.ligprep.sdf'
        logger.debug(f'Processed ligands in batch {smi} using ligprep')
        cmd = (f'{args.ligprep} -ma 500 -epik -bff 16 -ph 7.4 -pht 0.5 -max_stereo 1 -ac -g '
               f'-ismi {smi} -osd {sdf} -NJOBS {CPUS} -LOCAL -WAIT > {basename}.log')
        cmder.run(cmd, exit_on_error=True)
    else:
        logger.debug(f'Processed ligands in batch {smi} using gypsum_dl')
        sdf = f'{basename}.gypsum.sdf'
        cmd = (f'{args.gypsum} --source {smi} '
               f'--job_manager {"serial" if CPUS == 1 else "multiprocessing"} '
               f'--num_processors {CPUS} --max_variants_per_compound 1 > {basename}.gypsum.log')
        cmder.run(cmd, exit_on_error=True)
        cmder.run(f'mv gypsum_dl_success.sdf {sdf}')
        cmder.run(f'rm -f gypsum_dl_failed.smi')
    cmder.run(f'rm {smi}')
    logger.debug(f'Processed ligands in {smi} complete')
    return sdf
    
    
@task(inputs=batch_smiles, outputs=lambda i: i.replace('.batches.txt', '.sdf.txt'))
def prepare_sdf_ligand(inputs, output):
    sdfs = [smi_to_sdf(batch.strip()) for batch in read(inputs)]
    cmder.run(f'cat ligand.*.*.log > ligand.{"ligprep" if args.ligprep else "gypsum"}.log')
    cmder.run(f'rm ligand.*.*.log')
    cmder.run(f'rm {inputs}')
    write(output, sdfs)
    logger.debug(f'Successfully processed {len(sdfs):,} batches')
        

def calculate_descriptor(m):
    ds = Descriptors.CalcMolDescriptors(m)
    mw = ds.get('MolWt', 0)
    logP = ds.get('MolLogP', 0)
    tpsa = ds.get('TPSA', 20000)
    qed = ds.get('qed', 0)
    num_rotatable_bonds = ds.get('NumRotatableBonds', 10000)
    num_chiral_center = len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=True))
    hba = Descriptors.NOCount(m)
    hbd = Descriptors.NHOHCount(m)
    data = {'MW': mw, 'logP': logP, 'TPSA': tpsa, 'QED': qed,
            'rotatable_bound_number': num_rotatable_bonds, 'chiral_center': num_chiral_center,
            'HBA': hba, 'HBD': hbd}
    return data


def sep_sdf(sdf):
    logger.debug(f'  Separating ligands in {sdf} into individual SDF files')
    mol, descriptors = Chem.SDMolSupplier(sdf, removeHs=False), []
    for m in mol:
        if m:
            sd = f'{slugify(m.GetProp("_Name"))}.sdf'
            Chem.SDWriter(sd).write(m)
            des = calculate_descriptor(m)
            des['sdf'] = str(outdir / sd)
            descriptors.append(des)
    logger.debug(f'  Successfully saved {len(descriptors):,} ligands from {sdf} into individual SDF files')
    cmder.run(f'rm -f {sdf}')
    return descriptors
    
    
@task(inputs=prepare_sdf_ligand, outputs=lambda i: i.replace('.sdf.txt', '.sdf.parquet'))
def separate_sdf(inputs, output):
    sdfs, df = read(inputs), []
    with Pool(processes=args.cpu) as pool:
        for ds in pool.imap_unordered(sep_sdf, sdfs):
            df.extend(ds)

    if df:
        logger.debug(f'Successfully separated {len(df):,} ligands from {len(sdfs):,} batches into individual SDF file')
        df = pd.DataFrame(df)
        df.to_parquet(output, index=False)
        logger.debug(f'Successfully saved descriptors for {df.shape[0]:,} ligands')
    cmder.run(f'rm -f {inputs}')
    

def sdf_to_pdbqt(sdf):
    output = Path(sdf).with_suffix(".pdbqt")
    if not output.exists():
        p = cmder.run(f'mk_prepare_ligand.py -i {sdf} -o {output}', log_cmd=False)
        if p.returncode:
            output = ''
    return str(output)


@task(inputs=separate_sdf, outputs=[DESCRIPTORS])
def prepare_pdbqt_ligand(inputs, output):
    if args.pdbqt:
        df = utility.read(inputs, f'Failed to load sdf file path and descriptors from {inputs}')
        df['pdbqt'] = df.parallel_apply(lambda row: sdf_to_pdbqt(row.sdf), axis=1)
        utility.write(df, output)
        cmder.run(f'rm -f {inputs}')
        n = df[df['pdbqt'] != ''].shape[0]
        logger.debug(f'Successfully converted {n:,} ligands in SDF format to PDBQT format')
    else:
        cmder.run(f'mv {inputs} {output}')


def main():
    try:
        start = time.time()
        flow = Flow('ligand_preparation', short_description=__doc__.splitlines()[0], description=__doc__)
        flow.run(dry_run=args.dry, cpus=CPUS)
        t = str(timedelta(seconds=time.time() - start))
        utility.debug_and_exit(f'Ligand preparation complete in {t.split(".")[0]}\n', task=args.task, status=40)
    except Exception as e:
        utility.error_and_exit(f'Ligand preparation failed due to:\n{e}\n\n{traceback.format_exc()}', task=args.task,
                               status=-40)


if __name__ == '__main__':
    main()
