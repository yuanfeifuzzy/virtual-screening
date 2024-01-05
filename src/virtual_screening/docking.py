#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A command line tool for easily perform ligand-protein docking with variety docking software
"""

import json
import os
import sys
import argparse
import tempfile
import traceback
from pathlib import Path

import cmder
import MolIO
import vstool
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from rdkit.rdBase import DisableLog

_ = [DisableLog(level) for level in RDLogger._levels]

parser = argparse.ArgumentParser(prog='ligand-docking', description=__doc__.strip())
parser.add_argument('sdf', help="Path to a SDF file contains prepared ligands", type=vstool.check_file)
parser.add_argument('pdbqt', help="Path to prepared rigid receptor in .pdbqt format file", type=vstool.check_file)
parser.add_argument('--center', help="T X, Y, and Z coordinates of the center", type=float, nargs=3)
parser.add_argument('--flexible', help="Path to prepared flexible receptor file in .pdbqt format")
parser.add_argument('--size', help="The size in the X, Y, and Z dimension (Angstroms)",
                    type=int, nargs=3, default=[15, 15, 15])

parser.add_argument('--outdir', default='.', help="Path to a directory for saving output files",
                    type=vstool.mkdir)
parser.add_argument('--exe', help="Path to Uni-Dock executable", type=vstool.check_exe)

parser.add_argument('--task', type=int, default=0, help="ID associated with this task")

parser.add_argument('--name', help="Name of the docking job, default: %(default)s", default='docking')
parser.add_argument('--partition', help='Name of the queue, default: %(default)s', default='analysis')
parser.add_argument('--email', help='Email address for send status change emails')
parser.add_argument('--email-type', help='Email type for send status change emails, default: %(default)s',
                    default='ALL', choices=('BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'))
parser.add_argument('--day', type=int, default=0, help="Number of days the job needs to run, default: %(default)s")
parser.add_argument('--hold', action='store_true', help="Hold the submission without actually submit it to the queue")

parser.add_argument('--quiet', help='Process data quietly without debug and info message', action='store_true')
parser.add_argument('--verbose', help='Process data verbosely with debug and info message', action='store_true')
parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(quiet=args.quiet, verbose=args.verbose)

setattr(args, 'output', args.outdir / 'docking.sdf.gz')
setattr(args, 'scratch', vstool.mkdir(f'/scratch/{args.outdir.name}', task=args.task))


def ligand_list(sdf):
    output = Path(sdf).with_suffix('.txt')
    if output.exists():
        return output
    else:
        ligands, outdir = [], output.with_suffix('')
        outdir.mkdir(exist_ok=True)
        
        for i, ligand in enumerate(MolIO.parse_sdf(sdf)):
            if ligand.mol:
                ligands.append(ligand.sdf(output=outdir / f'{ligand.title}.sdf'))
            if args.debug and len(ligands) == 100:
                logger.debug(f'Debug mode enabled, only first 100 ligands passed filters in {sdf} were saved')
                break
        
        with output.open('w') as o:
            o.writelines(f'{ligand}\n' for ligand in ligands if ligand)
            logger.debug(f'Successfully saved {len(ligands):,} into individual SDF files')
        return output


def best_pose(sdf):
    ss = []
    for s in MolIO.parse(str(sdf)):
        if s.mol:
            ss.append(s)
    
    ss = sorted(ss, key=lambda x: x.score)[0] if ss else None
    return ss


def unidock(sdf):
    output = sdf.with_suffix('.docking.sdf.gz')
    if output.exists():
        logger.debug(f'Docking results for {sdf} already exists, skip re-docking')
    else:
        logger.debug(f'Docking ligands in {sdf} ...')
        batches = MolIO.split_sdf(sdf, prefix=f'{sdf.with_suffix("")}.', records=25000)
        gpu_id = int(sdf.name.split('.')[-2]) - 1
        (cx, cy, cz), (sx, sy, sz) = args.center, args.size
        
        poses = []
        for batch in batches:
            ligand_index, wd = ligand_list(batch), batch.with_suffix('')
            cmd = (f'export CUDA_VISIBLE_DEVICES={gpu_id} && {args.exe}',
                   f'--ligand_index {ligand_index}',
                   f'--receptor {args.pdbqt}',
                   f'--dir {wd}',
                   f'--center_x {cx} --center_y {cy} --center_z {cz} ',
                   f'--size_x {sx} --size_y {sy} --size_z {sz} ',
                   f'--search_mode balance',
                   '--scoring vina ',
                   '--verbosity 0',
                   f'&> /dev/null')
            
            p = cmder.run(' \\\n  '.join(cmd), fmt_cmd=False)
            if p.returncode:
                vstool.error_and_exit(f'Docking ligands in {batch} failed', task=args.task, status=-80)
            else:
                logger.debug(f'Parsing docking results in {wd}')
                n = 0
                for out in wd.glob(f'*_out.sdf'):
                    pose_out = best_pose(out)
                    if pose_out:
                        poses.append(pose_out)
                        n += 1
                logger.debug(f'Successfully parsed {n:,} docking poses for {batch}')
                cmder.run(f'rm -rf {wd}*')
        MolIO.write(poses, output)
        cmder.run(f'rm {sdf}')
    return output


def dock():
    wd = Path(tempfile.mkdtemp(prefix='docking.'))
    basename = wd / 'ligand.batch.'
    batches = MolIO.batch_sdf(args.sdf, 4, basename)
    sdfs = vstool.parallel_cpu_task(unidock, batches, processes=4)
    
    logger.debug(f'Merging and sorting docking results to {args.output}')
    MolIO.merge_sdf(sdfs, args.output, sort='ascending')
    
    if args.debug:
        logger.debug(f'Temporary directory: {wd} was not deleted due to debugging.\n')
    else:
        cmder.run(f'rm -rf {wd}')


def submit():
    (cx, cy, cz), (sx, sy, sz) = args.center, args.size
    directives = ['#SBATCH --nodelist=gpu', '#SBATCH --gpus-per-task=4']
    source = f'source {Path(vstool.check_exe("python")).parent / "activate"}'
    cd = f'cd {args.outdir} || {{ echo "Failed to cd into {args.outdir}!"; exit 1; }}'
    
    cmds = [nodelist, gpu_per_task, source, cd]
    cmd = ['vs-docking', args.sdf, args.pdbqt,
           f'--center {cx} {cy} {cz}', f'--size {sx} {sy} {sz}',
           f'--exe {args.exe.strip()}', f'--task {args.task}']
    
    if args.flexible:
        cmd.append(f'--flexible {args.flexible}')
    
    cmds.append(vstool.qvd(cmd, args))
    
    vstool.submit('\n'.join(cmds), job_name=args.name, day=args.day or 1, hour=0, partition=args.partition,
                  email=args.email, mail_type=args.email_type, log='%x.log', directives=directives,
                  script='docking.sh', hold=args.hold)


@vstool.profile(task=args.task, status=80, error_status=-80, task_name='Docking task')
def main():
    if args.output.exists():
        logger.debug(f'Docking result {args.output} already exists, skip re-docking')
    else:
        if args.day or args.hold:
            submit()
        else:
            vstool.task_update(args.task, 70)
            dock()


if __name__ == '__main__':
    main()
