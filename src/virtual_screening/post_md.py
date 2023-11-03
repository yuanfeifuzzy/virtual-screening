#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A command line tool for pst-docking analysis
"""

import argparse
from pathlib import Path

import MolIO
import cmder
import vstool
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(prog='post-md', description=__doc__.strip())
parser.add_argument('wd', help="Path to a directory contains md output", type=vstool.check_dir)
parser.add_argument('--summary', help='Path to a CSV file for saving MD summary results.')

parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
parser.add_argument('--version', version=vstool.get_version(__package__), action='version')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)


def parse_eaf(eaf, summary=None):
    logger.debug(f'Parsing {eaf} ...')
    wd = eaf.parent
    sdf, output = f'{wd}.sdf', f'{wd}.rmsd.csv'
    rmsd, flag, n = [], 0, 1
    with eaf.open() as f:
        for line in f:
            if "RMSD" in line:
                flag = 1
                n = 1
            elif flag == 1:
                if n == 0:
                    if 'FitBy = "(protein)"' in line:
                        flag = 2
                        n = 2
                    else:
                        flag = 0
                        n = 1
                else:
                    n -= 1
            elif flag == 2:
                if n == 0:
                    rmsd = line.strip().split("= [")[1].replace(" ]", "").split(" ")
                    break
                else:
                    n -= 1

    df = {}
    if rmsd:
        rmsd = np.array(rmsd, dtype=float)
        try:
            s = next(MolIO.parse_sdf(sdf))
            score = s.score if s else np.nan
        except Exception as e:
            score = np.nan
            logger.error(f'Failed to get docking score from {sdf} due to {e}')
        df = {'ligand': wd.name, 'score': score, 'rmsd': np.mean(rmsd),
              'rmsd_min': np.min(rmsd), 'rmsd_max': np.max(rmsd)}

        if summary:
            dd = pd.DataFrame([df])
            dd.to_csv(output, index=False, float_format='%.4f')
            dd.to_csv('rmsd.csv', index=False, float_format='%.4f')
            logger.debug(f'Successfully saved rmsd results to {output}')

            out = Path(summary).parent / f'{wd.name}.md.zip'
            cmder.run(f'zip -r {out} rmsd.csv md.eaf md-out.cms md_trj/', cwd=str(wd))
            if not args.debug:
                cmder.run(f'rm -r {wd}')
    return df


def main():
    df = (parse_eaf(eaf, args.summary) for eaf in args.wd.glob('**/md.eaf'))
    df = pd.DataFrame([d for d in df if d])
    if args.summary:
        df.to_csv(args.summary, index=False, float_format='%.4f')
        logger.debug(f'MD summary was successfully saved to {args.summary}')
    else:
        print(df)


if __name__ == '__main__':
    main()
