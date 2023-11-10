#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cluster poses using k-mean clusters
"""

import os
import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path
from datetime import timedelta
from multiprocessing import cpu_count

import vstool
import MolIO
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from sklearn.cluster import MiniBatchKMeans

parser = argparse.ArgumentParser(prog='cluster-pose', description=__doc__.strip())
parser.add_argument('sdf', help="Path to a SDF file contains docking poses", type=vstool.check_file)
parser.add_argument('--clusters', help="Number of clusters, default: %(default)s", default=100, type=int)
parser.add_argument('--method', help="Method for generating fingerprints, default: %(default)s",
                    default='morgan2', choices=('morgan2', 'morgan3', 'ap', 'rdk5'))
parser.add_argument('--bits', help="Number of fingerprint bits, default: %(default)s", default=1024, type=int)
parser.add_argument('--top', help="Percentage of top poses need to be retained for "
                                  "downstream analysis, default: %(default)s", type=float, default=10)
parser.add_argument('--output', help="Path to a SDF file for saving output poses",
                    default='cluster.pose.sdf')

args = parser.parse_args()
logger = vstool.setup_logger(verbose=True)
setattr(args, 'output', args.output or args.sdf.parent / 'cluster.pose.sdf')

if Path(args.output).exists():
    vstool.debug_and_exit(f'Cluster pose {args.output} already exists, skip re-clustering\n')


def generate_fingerprint(mol, bits=1024, method='rdk5'):
    mol, name = mol
    if method == 'morgan2':
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=bits)
    elif method == 'morgan3':
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=bits)
    elif method == 'ap':
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=bits)
    elif method == 'rdk5':
        fp = Chem.RDKFingerprint(mol, maxPath=5, fpSize=bits, nBitsPerHash=2)
    else:
        fp = []
        vstool.error_and_exit(f'Invalid fingerprint generate method {method}, cannot continue')

    a = np.zeros((1,), int)
    DataStructs.ConvertToNumpyArray(fp, a)
    return a, name


def kmeans_cluster(x, ligands, n_clusters=1000, batch_size=1024):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, n_init='auto')
    km = kmeans.fit(x)
    df = pd.DataFrame({'ligand': ligands, 'cluster': km.labels_})
    return df


def main():
    n = MolIO.count_sdf(str(args.sdf))
    if n <= args.clusters:
        logger.debug(f'Only {n} <= {args.clusters} poses were found in {args.sdf}, no clustering will be performed')
        with open(args.output, 'w') as o:
            o.writelines(pose.sdf(title=pose.title.rsplit('_', 1)[0]) for pose in MolIO.parse_sdf(args.sdf))
    else:
        fp, names = [], []
        with Chem.SDMolSupplier(str(args.sdf), removeHs=False) as f:
            mol = ((m, m.GetProp('_Name')) for m in f if m)

            processes = cpu_count()
            logger.debug(f'Generating fingerprints using {args.method} method with {processes} CPUs')
            fps = vstool.parallel_cpu_task(generate_fingerprint, mol,
                                           bits=args.bits, method=args.method, processes=processes)

        if fps:
            fp, names = [x[0] for x in fps], [x[1] for x in fps]
        logger.debug(f'Successfully generated fingerprints for {len(names)} molecules')

        logger.debug('Clustering top poses using scikit-learn k-mean mini-batches')
        dd = kmeans_cluster(np.array(fp), names, n_clusters=args.clusters, batch_size=100)
        logger.debug('Clustering top poses using scikit-learn k-mean mini-batches complete')

        logger.debug('Getting best pose in each cluster')
        dd[['ligand', 'score']] = dd['ligand'].str.rsplit('_', n=1, expand=True)
        dd['score'] = dd['score'].astype(float)
        dd = dd.sort_values(by=['cluster', 'score'])
        dd = dd.drop_duplicates(subset=['cluster'])

        out = f'{args.output}.tmp.sdf'
        # dd.to_csv(Path(output).with_suffix('.csv'), index=False)
        with Chem.SDMolSupplier(str(args.sdf), removeHs=False) as f, Chem.SDWriter(out) as o:
            mol = {m.GetProp('_Name'): m for m in f if m}
            for row in dd.itertuples():
                o.write(mol[f'{row.ligand}_{row.score}'])

        logger.debug('Sorting best pose in each cluster ...')
        ss = (s for s in MolIO.parse_sdf(out))
        ss = sorted([s for s in ss if s.mol], key=lambda x: x.score)
        with open(args.output, 'w') as o:
            o.writelines(s.sdf(title=s.title.rsplit('_', 1)[0]) for s in ss)
        os.unlink(out)
        logger.debug(f'Successfully saved {len(ss):,} poses to {args.output}')


if __name__ == '__main__':
    main()
