#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A command line tool for easily perform ligand-protein docking with variety docking software
"""
import gzip
import json
import os
import sys
import tempfile
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
parser.add_argument('sdf', help="Path to a SDF file contains prepared ligands", type=vstool.check_file)
parser.add_argument('pdbqt', help="Path to prepared rigid receptor in .pdbqt format file", type=vstool.check_file)
parser.add_argument('--center', help="The X, Y, and Z coordinates of the center", type=float, nargs=3)
parser.add_argument('--flexible', help="Path to prepared flexible receptor file in .pdbqt format")
parser.add_argument('--size', help="The size in the X, Y, and Z dimension (Angstroms)",
                    type=int, nargs=3, default=[15, 15, 15])
parser.add_argument('--output', help="Path to a directory for saving output files")

parser.add_argument('--quiet', help='Enable quiet mode.', action='store_true')
parser.add_argument('--verbose', help='Enable verbose mode.', action='store_true')
parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)
setattr(args, 'output', Path(args.output) if args.output else args.sdf.with_suffix('.docking.sdf.gz'))
setattr(args, 'outdir', vstool.mkdir(args.output.parent))


def batch_ligand(sdf):
    outdir = vstool.mkdir(sdf.with_suffix(''))
    batches = MolIO.split_sdf(str(sdf), outdir / 'batch.', records=2500)
    return batches, outdir


def ligand_list(sdf, debug=False):
    logger.debug(f'Parsing {sdf} into individual files')
    output = Path(sdf).with_suffix('.txt')
    ligands, outdir = [], vstool.mkdir(Path(sdf).with_suffix(''))

    for ligand in MolIO.parse_sdf(sdf):
        if ligand.mol:
            ligands.append(ligand.sdf(output=outdir / f'{ligand.title}.sdf'))
            if debug and len(ligands) == 100:
                break

    with output.open('w') as o:
        o.writelines(f'{ligand}\n' for ligand in ligands if ligand)
    logger.debug(f'Successfully saved {len(ligands):,} ligands into individual SDF files')
    return output, ligands


def unidock(batch):
    (cx, cy, cz), (sx, sy, sz) = args.center, args.size
    cmd = (f'/work/08944/fuzzy/share/software/Uni-Dock/bin/unidock --receptor {args.pdbqt} '
           f'--ligand_index {batch} --search_mode balance --scoring vina --max_gpu_memory 32000 --cpu 16 '
           f'--center_x {cx} --center_y {cy} --center_z {cz} --size_x {sx} --size_y {sy} --size_z {sz} '
           f'--dir {Path(batch).with_suffix("")} &> {Path(batch).with_suffix(".log")}')

    p = cmder.run(cmd)
    return p.returncode


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

    ss = sorted(ss, key=lambda x: x.score)[0] if ss else None
    return ss


def main():
    if args.output.exists():
        logger.debug(f'Docking output for {args.sdf} already exists')
    else:
        outdir, outs = Path(tempfile.mkdtemp(prefix='docking.')), []
        batches = MolIO.split_sdf(str(args.ligand), outdir / 'batch.', records=2500)
        keep = 0

        for i, sdf in enumerate(batches):
            indices, ligands = ligand_list(sdf, debug=args.debug)
            if ligands:
                logger.debug(f'Docking ligands in {sdf}')
                keep = unidock(indices)
                processes = min(len(ligands), 16)
                poses = vstool.parallel_cpu_task(best_pose, ligands, processes=processes)
                poses = (pose for pose in poses if pose)
                poses = sorted(poses, key=lambda x: x.score)

                if poses:
                    out = MolIO.write(poses, sdf.with_suffix('.docking.sdf'))
                    outs.append(out)
                    logger.debug(f'Successfully saved {len(poses):,} poses to {out}.')
                    if not keep:
                        cmder.run(f'rm -rf {sdf.with_suffix("")}')
                else:
                    logger.warning(f'No pose was saved to docking output for batch {sdf}')
            else:
                logger.warning(f'No ligands was found in batch {sdf}')

            if args.debug and i == 3:
                break

        opener = gzip.open if args.output.name.endswith('.gz') else open
        with opener(args.output, 'wt') as o:
            for out in outs:
                with open(out) as f:
                    o.write(f.read())
        logger.debug(f'Docking results for {args.sdf} was saved to {args.output}')

        if not args.debug and not keep:
            cmder.run(f'rm -r {outdir}')


if __name__ == '__main__':
    main()
