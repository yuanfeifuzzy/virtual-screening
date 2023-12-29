# Virtual Screening

A pipeline for smartly performing virtual screening

# Example
```shell
source /work/08944/fuzzy/share/software/virtual-screening/venv/bin/activate

virtual-screening \
  /work/08944/fuzzy/share/chemical_library/docking/MolPort.ligand.core.sdf.gz \
  /work/08944/fuzzy/share/receptor/dc1_f4293/dc1_f4293.pdb \
  5.860 4.130 6.280 \
  --outdir /work/08944/fuzzy/share/DC1 \
  --residue 476 \
  --time 50 \
  --email fei.yuan@bcm.edu \
  --project CHE23039
```

Successfully run the above commands will submit a job named `vs` to job queue for 
performing virtual screening. Output files will save to `/work/08944/fuzzy/share/DC1`. 
To fine tune the screening, see the detailed usage and options below.

## Usage
```shell
source /work/08944/fuzzy/share/software/virtual-screening/venv/bin/activate

$ virtual-screening -h
usage: virtual-screening [-h] --center [CENTER ...] [--flexible FLEXIBLE] [--filter FILTER] [--size [SIZE ...]] [--outdir OUTDIR] [--scratch SCRATCH] [--residue [RESIDUE ...]] [--top TOP]
                         [--clusters CLUSTERS] [--method {morgan2,morgan3,ap,rdk5}] [--bits BITS] [--time TIME] [--nodes NODES] [--project PROJECT] [--email EMAIL]
                         [--email-type {NONE,BEGIN,END,FAIL,REQUEUE,ALL}] [--delay DELAY] [--hold] [--separate] [--debug] [--version]
                         ligand pdb

A pipeline for perform virtual screening in an easy and smart way

positional arguments:
  ligand                Path to a directory contains prepared ligands in SDF format
  pdb                   Path to the receptor in PDB format file

options:
  -h, --help            show this help message and exit
  --center [CENTER ...]
                        The X, Y, and Z coordinates of the center
  --flexible FLEXIBLE   Path to prepared flexible receptor file in PDBQT format
  --filter FILTER       Path to a JSON file contains descriptor filters
  --size [SIZE ...]     The size in the X, Y, and Z dimension (Angstroms)
  --outdir OUTDIR       Path to a directory for saving output files
  --scratch SCRATCH     Path to the scratch directory, default: /scratch/08944/fuzzy
  --residue [RESIDUE ...]
                        Residue numbers that interact with ligand via hydrogen bond
  --top TOP             Percentage of top poses need to be retained for downstream analysis, default: None
  --clusters CLUSTERS   Number of clusters for clustering top poses, default: None
  --method {morgan2,morgan3,ap,rdk5}
                        Method for generating fingerprints
  --bits BITS           Number of fingerprint bits
  --time TIME           MD simulation time, default: None ns.
  --nodes NODES         Number of nodes, default: 8.
  --project PROJECT     The nmme of project you would like to be charged, default: CHE23039
  --email EMAIL         Email address for send status change emails
  --email-type {NONE,BEGIN,END,FAIL,REQUEUE,ALL}
                        Email type for send status change emails, default: ALL
  --delay DELAY         Hours need to delay running the job.
  --hold                Only generate submit script but hold for submitting
  --separate            Separate docking and MD into 2 steps
  --debug               Enable debug mode (for development purpose).
  --version             show program's version number and exit
```
