#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A command line tool for easily generating ligand batches
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

parser = argparse.ArgumentParser(prog='batch-ligand', description=__doc__.strip())
parser.add_argument('ligand', help="Path to a single SDF file contains prepared ligands", type=vstool.check_file)
parser.add_argument('-o', '--outdir', default='.', help="Path to a directory for saving output files",
                    type=vstool.mkdir)
parser.add_argument('-b', '--batch', type=int, help="Number of batches that the docking job will be split to, "
                                              "default: %(default)s", default=8)
parser.add_argument('-t', '--task', type=int, default=0, help="ID associated with this task")

parser.add_argument('--partition', help='Name of the queue, default: %(default)s', default='normal')
parser.add_argument('--email', help='Email address for send status change emails')
parser.add_argument('--submit', action='store_true', help="Submit the job to the queue for processing")
parser.add_argument('--hold', action='store_true',
                    help="Hold the submission without actually submit the job to the queue")

parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
vstool.setup_logger(verbose=True)


def submit():
    data = {'bin': Path(vstool.check_exe('python')).parent, 'outdir': args.outdir}
    env = ['source {bin}/activate', '', 'cd {outdir} || {{ echo "Failed to cd into {outdir}!"; exit 1; }}', '']
    cmd = ['batch-ligand', args.ligand, f'--outdir {args.outdir}', f'--batch {args.batch}',
           f'--task {args.task}']
    
    vstool.submit('\n'.join(env).format(**data) + f' \\\n  '.join(cmd), cpus_per_task=1,
                  job_name='batch-ligand', hour=0, minute=30,
                  partition=args.partition, email=args.email, mail_type=args.email_type, log='%x.%j.log',
                  script='batch.ligand.sh', hold=args.hold)


@vstool.profile(task=args.task, status=70, error_status=-70, task_name='Docking task')
def main():
    if args.submit or args.hold:
        submit()
    else:
        outputs = [args.outdir / f'batch.{i+1}.sdf' for i in range(args.batch)]
        if outputs and all(output.exists() for output in outputs):
            logger.debug(f'Ligand batches already exist, skip re-generating')
        else:
            MolIO.batch_sdf(args.ligand, args.batch, args.outdir / 'batch.')


if __name__ == '__main__':
    main()
