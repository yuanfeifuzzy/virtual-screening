#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A command line tool for easily perform ligand-protein docking with variety docking software
"""

import json
import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import timedelta

import cmder
import MolIO
import vstool
from loguru import logger
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from rdkit.rdBase import DisableLog

_ = [DisableLog(level) for level in RDLogger._levels]


parser = argparse.ArgumentParser(prog='docking', description=__doc__.strip())
parser.add_argument('ligand', help="Path to a SDF file contains prepared ligands", type=vstool.check_file)
parser.add_argument('receptor', help="Path to prepared rigid receptor in .pdbqt format file", type=vstool.check_file)
parser.add_argument('--center', help="The X, Y, and Z coordinates of the center", type=float, nargs='+')
parser.add_argument('--flexible', help="Path to prepared flexible receptor file in .pdbqt format")
parser.add_argument('--filter', help="Path to a JSON file contains descriptor filters")
parser.add_argument('--size', help="The size in the X, Y, and Z dimension (Angstroms)",
                    type=int, nargs='*', default=[15, 15, 15])
parser.add_argument('--exe', help="Path to docking program executable, default: %(default)s",
                    type=vstool.check_exe, default='/work/08944/fuzzy/share/software/Uni-Dock/bin/unidock')

parser.add_argument('--pdb', help="Path to a PDB file contains receptor structure", type=vstool.check_file)
parser.add_argument('--residue', nargs='*', type=int,
                    help="Residue numbers that interact with ligand via hydrogen bond")
parser.add_argument('--top', help="Percentage of top poses need to be retained for "
                                  "downstream analysis, default: %(default)s", type=float, default=10)
parser.add_argument('--clusters', help="Number of clusters for clustering top poses, "
                                       "default: %(default)s", type=int, default=1000)
parser.add_argument('--method', help="Method for generating fingerprints, default: %(default)s",
                    default='morgan2', choices=('morgan2', 'morgan3', 'ap', 'rdk5'))
parser.add_argument('--bits', help="Number of fingerprint bits, default: %(default)s", default=1024, type=int)
parser.add_argument('--schrodinger', help='Path to Schrodinger Suite root directory, default: %(default)s',
                        type=vstool.check_dir, default='/work/02940/ztan818/ls6/software/DESRES/2023.2')
parser.add_argument('--md', help='Path to md executable, default: %(default)s',
                        type=vstool.check_exe, default='/work/08944/fuzzy/share/software/virtual-screening/venv/lib/python3.11/site-packages/virtual_screening/desmond_md.sh')
parser.add_argument('--time', type=float, help="MD simulation time, default: %(default)s ns.")
parser.add_argument('--summary', help='Path to a CSV file for saving MD summary results.')

parser.add_argument('--nodes', type=int, default=0, help="Number of nodes, default: %(default)s.")
parser.add_argument('--email', help='Email address for send status change emails')
parser.add_argument('--email-type', help='Email type for send status change emails, default: %(default)s',
                    default='ALL', choices=('NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'))
parser.add_argument('--delay', help='Hours need to delay running the job.', type=int, default=0)

parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
vstool.setup_logger(verbose=True)


def filtering(sdf, filters):
    try:
        mol = next(Chem.SDMolSupplier(str(sdf), removeHs=False))
        if not mol:
            return
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


def ligand_list(sdf, filters=None, debug=False):
    logger.debug(f'Parsing {sdf} into individual files')
    output = Path(sdf).with_suffix('.txt')
    ligands, outdir = [], Path(sdf).parent

    for ligand in MolIO.parse_sdf(sdf):
        if ligand.mol:
            out = outdir / f'{ligand.title}.sdf'
            ligand.sdf(output=out)
            if filters:
                out = filtering(out, filters)
            if out:
                ligands.append(out)
            # if debug and len(ligands) == 100:
            #     logger.debug(f'Debug mode enabled, only first 100 ligands passed filters in {sdf} were saved')
            #     break

    with output.open('w') as o:
        o.writelines(f'{ligand}\n' for ligand in ligands)
    return output, ligands


def unidock(batch):
    logger.debug(f'Docking ligands in {batch} ...')
    (cx, cy, cz), (sx, sy, sz) = args.center, args.size
    cmd = (f'{args.exe} --receptor {args.receptor} '
           f'--ligand_index {batch} --search_mode balance --scoring vina '
           f'--center_x {cx} --center_y {cy} --center_z {cz} --size_x {sx} --size_y {sy} --size_z {sz} '
           f'--dir {Path(batch).parent} &> {Path(batch).with_suffix(".log")}')

    start = time.time()
    cmder.run(cmd)
    t = str(timedelta(seconds=time.time() - start))
    logger.debug(f'Docking ligands in {batch} complete in {t.split(".")[0]}.\n')


def best_pose(sdf):
    ss, out = [], Path(f'{Path(sdf).with_suffix("")}_out.sdf')
    if out.exists():
        for s in MolIO.parse(str(out)):
            if s.mol and s.score < 0:
                ss.append(s)
                
        if not args.debug:
            try:
                os.unlink(out)
            except Exception as e:
                logger.debug(f'Failed to delete files after parse best pose due to {e}')

    if not args.debug:
        try:
            os.unlink(sdf)
        except Exception as e:
            logger.debug(f'Failed to delete files after parse best pose due to {e}')

    ss = sorted(ss, key=lambda x: x.score)[0] if ss else None
    return ss


def main():
    if args.filter:
        filters = vstool.check_file(args.filter)
        with open(filters) as f:
            filters = json.load(f)
    else:
        filters = None
    
    batch, ligands = ligand_list(args.ligand, filters=filters, debug=args.debug)
    unidock(batch)
    
    logger.debug(f'Getting docking poses and scores ...')
    poses = vstool.parallel_cpu_task(best_pose, ligands)
    logger.debug(f'Getting docking poses and scores complete.')
    
    logger.debug(f'Sorting poses and scores ...')
    poses = (pose for pose in poses if pose)
    poses = sorted(poses, key=lambda x: x.score)
    logger.debug(f'Sorting {len(poses):,} poses with scores complete.')
    
    if poses:
        output = str(args.ligand.with_suffix('.docking.sdf'))
        logger.debug(f'Saving poses to {output} ...')
        MolIO.write(poses, output)
        MolIO.write(poses, Path(args.summary).parent / 'docking.sdf')
        logger.debug(f'Successfully saved {len(poses):,} poses to {output}.\n')


if __name__ == '__main__':
    main()
