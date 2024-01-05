#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A pipeline for perform virtual screening in an easy and smart way
"""

import os
import argparse
from pathlib import Path

import cmder
import vstool


def main():
    parser = argparse.ArgumentParser(prog='vs', description=__doc__.strip())
    parser.add_argument('sdf', help="Path to a SDF file contains prepared ligands", type=vstool.check_file)
    parser.add_argument('pdb', help="Path to a PDB file contains receptor structure", type=vstool.check_file)
    parser.add_argument('--center', help="The X, Y, and Z coordinates of the center (Angstrom)", nargs=3, type=float,
                        required=True)
    parser.add_argument('--filter', help="Path to a JSON file contains descriptor filters")
    parser.add_argument('--flexible', help="Path to a PDBQT file contains prepared flexible receptor structure")
    parser.add_argument('--size', help="The size in the X, Y, and Z dimension (Angstroms)", type=int, nargs=3,
                        default=[15, 15, 15])
    
    parser.add_argument('--outdir', help="Path to a directory for saving output files", default='.', type=vstool.mkdir)
    parser.add_argument('--docker', help="Path to Uni-Dock executable", type=vstool.check_exe)
    
    parser.add_argument('--residue', nargs='*', type=int,
                        help="Residue numbers that interact with ligand via hydrogen bond")
    parser.add_argument('--top', help="Percentage of top poses need to be retained for "
                                      "downstream analysis, default: %(default)s", type=float, default=0)
    parser.add_argument('--clusters', help="Number of clusters for clustering top poses, "
                                           "default: %(default)s", type=int, default=100)
    parser.add_argument('--time', type=float, default=0,
                        help="Molecule dynamics simulation time in nanosecond, default: %(default)s")
    
    parser.add_argument('--task', type=int, default=0, help="ID associated with this task")
    parser.add_argument('--name', help="Name of the virtual screening job, default: %(default)s", default='vs')
    parser.add_argument('--day', type=int, default=10, help="Number of days the job needs to run, default: %(default)s")
    parser.add_argument('--email', help='Email address for send status change emails')
    parser.add_argument('--email-type', help='Email type for send status change emails, default: %(default)s',
                        default='ALL', choices=('BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'))
    parser.add_argument('--delay', help='Hours need to delay running the job.', type=int, default=0)
    parser.add_argument('--hold', help='Only generate submit script but hold for submitting', action='store_true')
    
    parser.add_argument('--quiet', help='Enable quiet mode.', action='store_true')
    parser.add_argument('--verbose', help='Enable verbose mode.', action='store_true')
    parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
    parser.add_argument('--version', version=vstool.get_version(__package__), action='version')
    
    args = parser.parse_args()
    vstool.setup_logger(verbose=True)
    
    summary = args.outdir / 'docking.score.md.rmsd.csv'
    if summary.exists():
        vstool.info_and_exit(f'Virtual screening result {smmary} already exists, skip re-processing',
                              task=args.task, status=140)
    
    directives = ['#SBATCH --nodelist=gpu', '#SBATCH --gpus-per-task=4']
    source = f'source {Path(vstool.check_exe("python")).parent / "activate"}'
    cd = f'cd {args.outdir} || {{ echo "Failed to cd into {args.outdir}!"; exit 1; }}'
    
    cmds = [source, cd]
    if args.filter:
        sdf = args.outdir / 'ligands.sdf'
        cmds.append(vstool.qvd(['vs-ligand-filter', args.sdf, args.filter, f'--output {sdf}'], args))
    else:
        sdf = args.sdf
    
    (cx, cy, cz), (sx, sy, sz) = args.center, args.size
    cmd = ['vs-docking', sdf, f'{args.pdb}qt',
           f'--center {cx} {cy} {cz}', f'--size {sx} {sy} {sz}',
           f'--exe {args.docker}', f'--task {args.task}']
    if args.flexible:
        cmd.append(f'--flexible {args.flexible}')
    cmds.append(vstool.qvd(cmd, args))

    if args.top or args.residue:
        cmd = ['vs-postdoc', str(args.outdir / 'docking.sdf.gz'),
               str(args.pdb), f'--top {args.top}', f'--clusters {args.clusters} ', f'--task {args.task}']
        if args.residue:
            cmd.append(f'--residue {" ".join(str(x) for x in args.residue)}')
        cmds.append(vstool.qvd(cmd, args))
        
        if args.time:
            cmd = ['vs-simulate', str(args.outdir / 'cluster.pose.sdf'), str(args.pdb),
                   f'--outdir {args.outdir}', f'--summary {summary.name}',
                   f'--time {args.time} ', f'--task {args.task}']
            cmds.append(vstool.qvd(cmd, args))
        else:
            vstool.info_and_exit(f'MD task will not run due to time is not set')
    else:
        vstool.info_and_exit(f'Post docking job will not run due to neither top no residue is set')
    
    vstool.submit('\n'.join(cmds), cpus_per_task=64,
                  job_name=args.name or 'vs', day=args.day, hour=0,
                  directives=directives, partition='analysis',
                  email=args.email, mail_type=args.email_type, log='%x.%j.log',
                  script=args.outdir / 'vs.sh', delay=args.delay, hold=args.hold)


if __name__ == '__main__':
    main()
