#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A command line tool for easy ligand-protein docking with variety docking software
"""

import json
import os
import sys
import argparse
import traceback
from pathlib import Path

import cmder
import MolIO
import vstool
from loguru import logger

parser = argparse.ArgumentParser(prog='ligand-docking', description=__doc__.strip())
parser.add_argument('ligand', help="Path to a directory contains prepared ligands in SDF format",
                    type=vstool.check_file)
parser.add_argument('receptor', help="Path to prepared rigid receptor in .pdbqt format file", type=vstool.check_file)
parser.add_argument('-g', '--grid', help="Path to prepared receptor maps filed file in .maps.fld format")
parser.add_argument('-f', '--flexible', help="Path to prepared flexible receptor file in .pdbqt format")
parser.add_argument('-y', '--filter', help="Path to a JSON file contains descriptor filters")
parser.add_argument('-s', '--size', help="The size in the X, Y, and Z dimension (Angstroms)", type=int, nargs='+')
parser.add_argument('-c', '--center', help="T X, Y, and Z coordinates of the center", type=float, nargs='+')

parser.add_argument('-o', '--outdir', default='.', help="Path to a directory for saving output files",
                    type=vstool.mkdir)
parser.add_argument('-e', '--exe', help="Path to docking program executable", type=vstool.check_exe)

parser.add_argument('--scratch', help="Path to the scratch directory, default: %(default)s",
                    default=Path(os.environ.get('SCRATCH', '/scratch')))
parser.add_argument('--cpu', type=int, default=0,
                    help="Maximum number of CPUs can be used for parallel processing, default: %(default)s")
parser.add_argument('--gpu', type=int, default=0,
                    help="Maximum number of GPUs can be used for parallel processing, default: %(default)s")

parser.add_argument('--batch', type=int, help="Number of batches that the docking job will be split to, "
                                              "default: %(default)s", default=8)
parser.add_argument('--task', type=int, default=0, help="ID associated with this task")

parser.add_argument('--name', help="Name of the docking job, default: %(default)s", default='docking')
parser.add_argument('--partition', help='Name of the queue, default: %(default)s', default='normal')
parser.add_argument('--email', help='Email address for send status change emails')
parser.add_argument('--email-type', help='Email type for send status change emails, default: %(default)s',
                    default='ALL', choices=('NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'))
parser.add_argument('--hour', type=int, default=12, help="Number of hours each batch needs to  "
                                                         "run, default: %(default)s")
parser.add_argument('--day', type=int, default=0, help="Number of days each batch needs to  "
                                                       "run, default: %(default)s")
parser.add_argument('--submit', action='store_true', help="Submit the job to the queue for processing")
parser.add_argument('--hold', action='store_true',
                    help="Hold the submission without actually submit the job to the queue")

parser.add_argument('-q', '--quiet', help='Process data quietly without debug and info message', action='store_true')
parser.add_argument('-v', '--verbose', help='Process data verbosely with debug and info message', action='store_true')
parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
vstool.setup_logger(quiet=args.quiet, verbose=args.verbose)

setattr(args, 'result', args.outdir / 'docking.sdf')
setattr(args, 'scratch', vstool.mkdir(args.scratch / f'{args.outdir.name}', task=args.task))
setattr(args, 'cpu', 4 if args.debug else vstool.get_available_cpus(args.cpu))
setattr(args, 'gpu', 2 if args.debug else len(vstool.get_available_gpus(args.gpu, task=args.task)))


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


def ligand_list(sdf, filters=None, debug=False):
    output = Path(sdf).with_suffix('.txt')
    if output.exists():
        return output
    else:
        ligands, outdir = [], Path(sdf).with_suffix('')
        outdir.mkdir(exist_ok=True)

        for ligand in MolIO.parse_sdf(sdf):
            out = outdir / f'{ligand.title}.sdf'
            ligand.sdf(output=out)
            if filters:
                out = filtering(out, filters)
            if out:
                ligands.append(out)
            if debug and len(ligands) == 100:
                logger.debug(f'Debug mode enabled, only first 100 ligands passed filters in {sdf} were saved')
                break

        with output.open('w') as o:
            o.writelines(f'{ligand}\n' for ligand in ligands)
        return output


def autodock():
    pass


def unidock(batch):
    logger.debug(f'Docking ligands in {batch} ...')
    gpu_id, log, outdir = batch.name.split('.')[-2], batch.with_suffix(".log"), batch.with_suffix('')
    gpu_id = 3
    (cx, cy, cz), (sx, sy, sz) = args.center, args.size
    cmd = (f'{args.exe} --receptor {args.receptor} '
           f'--ligand_index {batch} --devnum {gpu_id} --search_mode balance --scoring vina '
           f'--center_x {cx} --center_y {cy} --center_z {cz} --size_x {sx} --size_y {sy} --size_z {sz} '
           f'--dir {outdir} &> {log}')

    p = cmder.run(cmd, log_cmd=args.verbose)
    if p.returncode:
        utility.error_and_exit(f'Docking ligands in {batch} failed', task=args.task, status=-80)
    logger.debug(f'Docking ligands in {batch} complete.\n')
    return outdir


def gina(sdf):
    pass


def best_pose(sdf):
    ss = []
    for s in MolIO.parse(str(sdf)):
        if s.mol:
            ss.append(s)

    ss = sorted(ss, key=lambda x: x.score)[0] if ss else None
    return ss


def dock():
    output = args.ligand.with_suffix('.docking.sdf')
    basename = args.scratch / args.ligand.with_suffix('').name
    batches = [Path(f'{basename}.{i + 1}.sdf') for i in range(args.gpu)]
    batch_outputs = [Path(f'{basename}.{i + 1}.docking.sdf') for i in range(args.gpu)]
    
    if output.exists():
        logger.debug(f'Docking results for {args.ligand} already exists, skip re-docking')
    else:
        poses = []
        if batch_outputs and all(out.exists() for out in batch_outputs):
            for out in batch_outputs:
                logger.debug(f'Loading poses from {out} ...')
                for pose in MolIO.parse(str(out)):
                    if pose.mol and pose.score:
                        poses.append(pose)
        else:
            exe = args.exe.lower()
            if 'unidock' in exe:
                docker = unidock
            elif 'gina' in exe:
                docker = gina
            elif 'autodock' in exe:
                docker = autodock
            else:
                vstool.error_and_exit(f'Unknown exe {args.exe}, cannot continue', task=args.task, status=-70)

            outdir = args.scratch / args.ligand.with_suffix('').name
            MolIO.batch_sdf(args.ligand, args.gpu, f'{outdir}.')

            if args.filter:
                filters = vstool.check_file(args.filters)
                with open(filters) as f:
                    filters = json.load(f)
            else:
                filters = None
                
            batches = [ligand_list(batch, filters=filters, debug=args.debug) for batch in batches]

            vstool.parallel_gpu_task(docker, batches)

            for batch in batches:
                logger.debug(f'Parsing poses from {batch} ...')
                ps = vstool.parallel_cpu_task(best_pose, batch.with_suffix('').glob('*_out.sdf'))
                poses.extend(ps)
                MolIO.write(ps, str(batch.with_suffix('.docking.sdf')))
                logger.debug(f'Successfully parsed {len(ps):,} poses.')
                if args.debug:
                    cmder.run(f'rm -r {batch.with_suffix("")}*')

        poses = sorted(poses, key=lambda x: x.score)
        logger.debug(f'Saving poses to {output} ...')
        MolIO.write(poses, output)
        logger.debug(f'Successfully saved {len(poses):,} poses to {output}.\n')

    done, running, batches = [], [], []
    for batch in args.ligand.parent.glob('*.sdf'):
        if not batch.name.endswith('.docking.sdf'):
            batches.append(batch)
            out = batch.with_suffix('.docking.sdf')
            if out.exists():
                done.append(out)
            else:
                running.append(out)

    if running:
        logger.debug(f'The following {len(running)} batches are under processing or pending for processing:')
        for run in running:
            logger.debug(f'  {run}')
    else:
        logger.debug('All batches were processed')
        MolIO.merge_sdf(done, str(args.result))
    
        if args.debug:
            for batch in batches + done:
                os.unlink(batch)
            cmder.run(f'rm -r {args.scratch}')


def submit():
    outputs = list(args.outdir.glob('batch.*.docking.sdf'))
    if outputs and all(output.exist() for output in outputs):
        logger.debug(f'Docking results already exist')
    else:
        MolIO.batch_sdf(args.ligand, args.batch, args.outdir / 'batch.')

        data = {
            'bin': Path(vstool.check_exe('python')).parent, 'outdir': args.outdir,
            'exe': vstool.check_exe(args.exe),
            'ligand': args.outdir / f'batch."${{SLURM_ARRAY_TASK_ID}}".sdf',
            'receptor': args.receptor,
            'size': f'{args.size[0]} {args.size[1]} {args.size[2]}',
            'center': f'{args.center[0]} {args.center[1]} {args.center[2]}',
            'scratch': args.scratch, 'cpu': args.cpu, 'gpu': args.gpu
        }
        env = ['source {bin}/activate', '', 'cd {outdir} || {{ echo "Failed to cd into {outdir}!"; exit 1; }}', '']
        cmd = ['docking', str(args.ligand), str(args.receptor),
               f'--size {args.size[0]} {args.size[1]} {args.size[2]}',
               f'--center {args.center[0]} {args.center[1]} {args.center[2]}',
               f'--cpu {args.cpu}', f'--gpu {args.gpu}', f'--scratch {args.scratch}',
               f'--exe {args.exe.strip()}', f'--task {args.task}']

        if args.flexible:
            cmd.append(f'--flexible {args.flexible}')

        if args.filter:
            cmd.append(f'--filter {args.filter}')

        if args.verbose:
            cmd.append('--verbose')

        if args.quiet:
            cmd.append('--quiet')

        if args.debug:
            cmd.append('--debug')

        vstool.submit('\n'.join(env).format(**data) + f' \\\n  '.join(cmd).format(**data),
                      cpus_per_task=args.cpu, gpus_per_task=args.gpu, job_name=args.name,
                      day=args.day, hour=args.hour, array=f'1-{args.batch}', partition=args.partition,
                      email=args.email, mail_type=args.email_type, log='%x.%j.%A.%a.log',
                      script='docking.sh', hold=args.hold)


@vstool.profile(task=args.task, status=70, error_status=-70, task_name='Docking task')
def main():
    if args.result.exists():
        logger.debug(f'Docking result {args.result} already exists, skip re-docking')
    else:
        if args.submit or args.hold:
            submit()
        else:
            dock()


if __name__ == '__main__':
    main()
