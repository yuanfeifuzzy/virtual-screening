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

import vstool
import MolIO
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors
from sklearn.cluster import MiniBatchKMeans

RDLogger.DisableLog('rdApp.*')
logger = vstool.setup_logger()


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

    a = np.zeros((1, ), int)
    DataStructs.ConvertToNumpyArray(fp, a)
    return a, name


def kmeans_cluster(x, ligands, n_clusters=1000, batch_size=1024):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, n_init='auto')
    km = kmeans.fit(x)
    df = pd.DataFrame({'ligand': ligands, 'cluster': km.labels_})
    return df


def clustering(sdf, output='cluster.pose.sdf', n_clusters=1000, method='rdk5', bits=1024, processes=32,
               quiet=False, verbose=False):
    vstool.setup_logger(quiet=quiet, verbose=verbose)

    fp, names = [], []
    with Chem.SDMolSupplier(sdf) as f:
        # By default, pickling removes mol properties
        mol = ((m, m.GetProp('_Name')) for m in f if m)
        if processes > 1:
            logger.debug(f'Generating fingerprints using {method} method with {processes} CPUs')
            fps = vstool.parallel_cpu_task(generate_fingerprint, mol, bits=bits, method=method, chunksize=500,
                                           processes=32)
        else:
            logger.debug(f'Generating fingerprints using {method} method with a single CPU')
            fps = [generate_fingerprint(m, bits=bits, method=method) for m in mol]
    if fps:
        fp, names = [x[0] for x in fps], [x[1] for x in fps]
    logger.debug(f'Successfully generated fingerprints for {len(names)} molecules')
    
    logger.debug('Clustering top poses using scikit-learn k-mean mini-batches')
    dd = kmeans_cluster(np.array(fp), names, n_clusters=n_clusters, batch_size=256*processes)
    logger.debug('Clustering top poses using scikit-learn k-mean mini-batches complete')
    
    logger.debug('Getting best pose in each cluster')
    dd[['ligand', 'score']] = dd['ligand'].str.rsplit('_', n=1, expand=True)
    dd['score'] = dd['score'].astype(float)
    dd = dd.sort_values(by=['cluster', 'score'])
    dd = dd.drop_duplicates(subset=['cluster'])
    
    if output:
        out = f'{output}.tmp.sdf'
        dd.to_csv(Path(output).with_suffix('.csv'), index=False)
        with Chem.SDMolSupplier(str(sdf), removeHs=False) as f, Chem.SDWriter(out) as o:
            mol = {m.GetProp('_Name'): m for m in f if m}
            for row in dd.itertuples():
                o.write(mol[f'{row.ligand}_{row.score}'])

        logger.debug('Sorting best pose in each cluster ...')
        ss = (s for s in MolIO.parse_sdf(out))
        ss = sorted([s for s in ss if s.mol], key=lambda x: x.score)
        with open(output, 'w') as o:
            o.writelines(s.sdf(title=s.title.rsplit('_', 1)[0]) for s in ss)
        os.unlink(out)
        logger.debug(f'Successfully saved {len(ss):,} poses to {output}')
    return dd


def main():
    parser = argparse.ArgumentParser(prog='cluster-pose', description=__doc__.strip())
    parser.add_argument('path', help="Path to a parquet file contains docking scores")
    parser.add_argument('-o', '--output', help="Path to a output for saving best poses in each cluster "
                                               "in SDF format, default: %(default)s", default='cluster.pose.sdf')
    parser.add_argument('-n', '--clusters', help="Number of clusters, default: %(default)s", default=1000, type=int)
    parser.add_argument('-m', '--method', help="Method for generating fingerprints, default: %(default)s",
                        default='morgan2', choices=('morgan2', 'morgan3', 'ap', 'rdk5'))
    parser.add_argument('-b', '--bits', help="Number of fingerprint bits, default: %(default)s", default=1024, type=int)
    parser.add_argument('-c', '--cpu', default=32, type=int,
                        help='Number of maximum processors (CPUs) can be use for processing data')
    parser.add_argument('-q', '--quiet', help='Process data quietly without debug and info message',
                        action='store_true')
    parser.add_argument('-v', '--verbose', help='Process data verbosely with debug and info message',
                        action='store_true')
    
    parser.add_argument('--wd', help="Path to work directory", default='.')
    parser.add_argument('--task', type=int, default=0, help="ID associated with this task")
    parser.add_argument('--submit', action='store_true', help="Submit job to job queue instead of directly running it")
    parser.add_argument('--hold', action='store_true',
                        help="Hold the submission without actually submit the job to the queue")

    args = parser.parse_args()
    vstool.setup_logger(quiet=args.quiet, verbose=args.verbose)
    
    try:
        start = time.time()
        output = Path(args.output) or Path(sdf).resolve().parent / 'cluster.pose.sdf'
        if output.exists():
            vstool.debug_and_exit(f'Cluster pose already exists, skip re-processing\n', task=args.task, status=105)

        clustering(args.path, n_clusters=args.clusters, method=args.method, bits=args.bits, output=str(output),
                   processes=args.cpu, quiet=args.quiet, verbose=args.verbose)
        t = str(timedelta(seconds=time.time() - start))
        vstool.debug_and_exit(f'Cluster top pose complete in {t.split(".")[0]}\n', task=args.task, status=105)
    except Exception as e:
        vstool.error_and_exit(f'Cluster top pose failed due to\n{e}\n\n{traceback.format_exc()}\n',
                               task=args.task, status=-105)


if __name__ == '__main__':
    main()
