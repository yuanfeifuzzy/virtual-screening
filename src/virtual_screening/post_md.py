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
parser.add_argument('summary', help='Path a CSV file for saving MD summary results.')

parser.add_argument('--quiet', help='Enable quiet mode.', action='store_true')
parser.add_argument('--verbose', help='Enable verbose mode.', action='store_true')
parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)


def main():
    df = []
    for x in args.wd.glob('*.rmsd.csv'):
        df.append(pd.read_csv(x))

    if df:
        df = pd.concat(df)
        columns = ['ligand', 'score']
        df = df[columns + [c for c in df.columns if c not in columns]]
        df.to_csv(args.summary, index=False, float_format='%.4f')
        print(df)
        logger.debug(f'MD summary was successfully saved to {args.summary}')

    logger.debug('Mission accomplished')


if __name__ == '__main__':
    main()
