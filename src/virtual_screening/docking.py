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
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from rdkit.rdBase import DisableLog

_ = [DisableLog(level) for level in RDLogger._levels]

parser = argparse.ArgumentParser(prog='docking', description=__doc__.strip())
parser.add_argument('ligand', help="Path to a SDF file contains prepared ligands", type=vstool.check_file)
parser.add_argument('pdbqt', help="Path to prepared rigid receptor in .pdbqt format file", type=vstool.check_file)
parser.add_argument('--center', help="The X, Y, and Z coordinates of the center", type=float, nargs='+')
parser.add_argument('--flexible', help="Path to prepared flexible receptor file in .pdbqt format")
parser.add_argument('--filter', help="Path to a JSON file contains descriptor filters")
parser.add_argument('--size', help="The size in the X, Y, and Z dimension (Angstroms)",
                    type=int, nargs='*', default=[15, 15, 15])
parser.add_argument('--outdir', help="Path to a directory for saving output files", type=vstool.mkdir)

parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)


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
    ligands, outdir = [], vstool.mkdir(Path(sdf).with_suffix(''))

    for ligand in MolIO.parse_sdf(sdf):
        if ligand.mol:
            out = ligand.sdf(output=outdir / f'{ligand.title}.sdf')
            if filters:
                out = filtering(out, filters)
            if out:
                ligands.append(out)
                if debug and len(ligands) == 100:
                    logger.debug(f'Debug mode enabled, only first 100 ligands passed filters in {sdf} were saved')
                    break

    with output.open('w') as o:
        o.writelines(f'{ligand}\n' for ligand in ligands)
    logger.debug(f'Successfully saved {len(ligands):,} ligands into individual SDF files')
    return output, ligands


def unidock(batch):
    (cx, cy, cz), (sx, sy, sz) = args.center, args.size
    cmd = (f'/work/08944/fuzzy/share/software/Uni-Dock/bin/unidock --receptor {args.pdbqt} '
           f'--ligand_index {batch} --search_mode balance --scoring vina --max_gpu_memory 32000 --cpu 16 '
           f'--center_x {cx} --center_y {cy} --center_z {cz} --size_x {sx} --size_y {sy} --size_z {sz} '
           f'--dir {Path(batch).with_suffix("")} &> {Path(batch).with_suffix(".log")}')

    cmder.run(cmd)


def best_pose(sdf):
    ss, out = [], Path(f'{Path(sdf).with_suffix("")}_out.sdf')
    if out.exists():
        try:
            for s in MolIO.parse(str(out)):
                if s and s.mol and s.score < 0:
                    ss.append(s)
        except Exception as e:
            logger.error(f'Failed to parse {sdf} due to {e}')
    if not args.debug:
        cmder.run(f'rm -f {sdf} {out}', log_cmd=False)
        # try:
        #     for s in MolIO.parse(str(out)):
        #         if s and s.mol and s.score < 0:
        #             ss.append(s)
    #         os.unlink(out)
    #     except Exception as e:
    #         logger.debug(f'Failed to parse best pose in {out} due to {e}')
    #
    # if not args.debug:
    #     try:
    #         os.unlink(sdf)
    #     except Exception as e:
    #         logger.debug(f'Failed to delete {sdf} after parse best pose due to {e}')

    ss = sorted(ss, key=lambda x: x.score)[0] if ss else None
    return ss


def main():
    start = time.time()
    if args.filter:
        with open(vstool.check_file(args.filter)) as f:
            filters = json.load(f)
    else:
        filters = None
    
    batch, ligands = ligand_list(args.ligand, filters=filters, debug=args.debug)

    logger.debug(f'Docking {len(ligands):,} ligands in {batch} ...')
    s = time.time()
    unidock(batch)
    t = str(timedelta(seconds=time.time() - s))
    logger.debug(f'Docking {len(ligands):,} ligands in {batch} complete in {t.split(".")[0]}.\n')
    
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
        MolIO.write(poses, args.outdir / 'docking.sdf')
        logger.debug(f'Successfully saved {len(poses):,} poses to {output}.\n')
    
    t = str(timedelta(seconds=time.time() - start))
    logger.debug(f'Docking and parse score for {args.ligand} complete in {t.split(".")[0]}.\n')


if __name__ == '__main__':
    main()
