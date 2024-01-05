#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filtering ligands by apply filters from a JSON file
"""

import json
import os
import sys
import argparse
import traceback
from pathlib import Path
from itertools import islice

import cmder
import MolIO
import vstool
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from rdkit.rdBase import DisableLog

_ = [DisableLog(level) for level in RDLogger._levels]
logger = vstool.setup_logger()


def _filtering(ss, filters=None):
    if not ss.mol:
        return
    try:
        supplier = Chem.SDMolSupplier()
        supplier.SetData(ss.s)
        mol = supplier[0]
        # mol = Chem.AddHs(supplier[0]) # This will lead to more molecules with high rotatable bonds
        # mol = Chem.MolFromMolBlock(ss.mol, removeHs=False) # This object is not pickleable
        title = ss.title
    except Exception as e:
        logger.error(f'Failed to read the following block due to \n{e}\n\n{traceback.format_exc()}:\n\n{ss.s}')
        return
    
    mw = Descriptors.MolWt(mol)
    if mw < filters['min_mw'] or mw > filters['max_mw']:
        logger.debug(f"{title}: MW={mw:.4f} out of range [{filters['min_mw']}, {filters['max_mw']}]")
        return
    
    hba = Descriptors.NOCount(mol)
    if hba > filters['hba']:
        logger.debug(f'{title}: hba = {hba} > {filters["hba"]}')
        return
    
    hbd = Descriptors.NHOHCount(mol)
    if hbd > filters['hbd']:
        logger.debug(f'{title}: hbd = {hbd} > {filters["hbd"]}')
        return
    
    logP = Descriptors.MolLogP(mol)
    if logP < filters['min_logP'] or logP > filters['max_logP']:
        logger.debug(f"{title}: logP = {logP:.4f} out of range [{filters['min_logP']}, {filters['max_logP']}]")
        return
    
    # https://www.rdkit.org/docs/GettingStartedInPython.html#lipinski-rule-of-5
    lipinski_violation = sum([mw > 500, hba > 10, hbd > 5, logP > 5])
    if lipinski_violation > filters['lipinski_violation']:
        logger.debug(f'{title}: lipinski_violation = {lipinski_violation} > {filters["lipinski_violation"]}')
        return
    
    ds = Descriptors.CalcMolDescriptors(mol)
    num_chiral_center = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=True))
    if num_chiral_center > filters["chiral_center"]:
        logger.debug(f'{title}: num_chiral_center = {num_chiral_center} > {filters["chiral_center"]}')
        return
    
    rbn = ds.get('NumRotatableBonds', 0)
    if rbn > filters['rotatable_bound_number']:
        logger.debug(f'{title}: rotatable_bound_number = {rbn} > {filters["rotatable_bound_number"]}')
        return
    
    tpsa = ds.get('TPSA', 0)
    if tpsa > filters['tpsa']:
        logger.debug(f'{title}: TPSA = {tpsa} > {filters["TPSA"]}')
        return
    
    qed = ds.get('qed', 0)
    if qed < filters['qed']:
        logger.debug(f'{title}: QED = {qed:.4f} < {filters["qed"]}')
        return
    
    return ss


def filtering(sdf, filters, outdir=None, output=None, cpu=8, quiet=False, verbose=False, debug=False):
    vstool.setup_logger(verbose=verbose, quiet=quiet)
    if outdir or output:
        with open(filters) as f:
            filters = json.load(f)
        ss = MolIO.parse_sdf(sdf)
        if debug:
            logger.debug(f'Filtering first 100 ligands [DEBUG]')
            ss = islice(ss, 100)
        outs = vstool.parallel_cpu_task(_filtering, ss, filters=filters, processes=cpu)
        MolIO.write(outs, outdir=outdir, output=output)
    else:
        raise ValueError('Neither outdir nor output was provided to save filtering results')
    

def main():
    parser = argparse.ArgumentParser(prog='vs-filter-ligand', description=__doc__.strip())
    parser.add_argument('sdf', help="Path to a SDF file contains prepared ligands", type=vstool.check_file)
    parser.add_argument('filters', help="Path to JSON file contains filters", type=vstool.check_file)
    parser.add_argument('--outdir', help="Path to a directory for saving individual SDF files passed filters")
    parser.add_argument('--output', help="Path to a SDF file for saving ligands passed filters")
    
    parser.add_argument('--cpu', type=int, default=8,
                        help="Maximum number of CPUs can be used for parallel processing, default: %(default)s")
    
    parser.add_argument('--quiet', help='Process data quietly without debug and info message',
                        action='store_true')
    parser.add_argument('--verbose', help='Process data verbosely with debug and info message',
                        action='store_true')
    parser.add_argument('--debug', help='Enable debug mode (for development purpose).', action='store_true')
    parser.add_argument('--version', version=vstool.get_version(__package__), action='version')
    
    args = parser.parse_args()
    filtering(args.sdf, args.filters, outdir=args.outdir, output=args.output, quiet=args.quiet, verbose=args.verbose,
              debug=args.debug)


if __name__ == '__main__':
    main()
