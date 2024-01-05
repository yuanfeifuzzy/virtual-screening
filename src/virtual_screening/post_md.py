#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A command line tool for pst-docking analysis
"""

import argparse
import os
from pathlib import Path

import MolIO
import cmder
import vstool
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(prog='post-md', description=__doc__.strip())
parser.add_argument('wd', help="Path to a directory contains md output", type=vstool.check_dir)
parser.add_argument('summary', help='Basename of a CSV file for saving MD summary results.')

parser.add_argument('--scratch', help='The path to scratch directory that contains intermediate files '
                                      'need to be deleted.', type=vstool.check_dir)
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)


def main():
    df = []
    for x in args.wd.glob('*.rmsd.csv'):
        df.append(pd.read_csv(x))
        if args.scratch:
            os.unlink(x)
    if df:
        df = pd.concat(df).sort_values(by='score')
        columns = ['ligand', 'score']
        df = df[columns + [c for c in df.columns if c not in columns]]
        df.to_csv(args.wd / args.summary, index=False, float_format='%.4f')
        print(df)
        logger.debug(f'MD summary was successfully saved to {args.wd / args.summary}')

    if args.scratch:
        logger.debug(f'Cleaning up directory {args.scratch} that contains intermediate files')
        cmder.run(f'rm -r {args.scratch}')

    logger.debug('Mission accomplished')


if __name__ == '__main__':
    main()
