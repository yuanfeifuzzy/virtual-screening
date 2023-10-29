#! /bin/bash
# -*- coding:utf-8 _*-
# @author: Lu Chong
# @file: desmond_md.sh
# @time: 2023/8/14/13:24

read -r -d '' usage << EOF
Molecular dynamics simulation using Desmond

Usage:

  desmond_md.sh wd complex time

    - wd         Path to work directory
    - complex    Ligand and receptor complex in mae format
    - time       Simulation time in nanosecond
EOF

if [[ $# == 0 ]]
then
  echo "${usage}"
  exit 0
else
  if [[ $# != 3 ]]
  then
    echo "${usage}"
    exit 1
  fi
fi

workdir=${1}
in_mae_complex=${2}
md_time=${3}

md_msj=$workdir/md.msj
md_cfg=$workdir/md.cfg
md_out=$workdir/md_out.cms
sys_build_out=$workdir/sys.cms

sys_build_msj=$workdir/setup.msj
sys_build_out=$workdir/sys.cms

mkdir -p "$workdir"
cd "$workdir" || exit 1

Desmond="${Desmond:=/work/08944/fuzzy/share/software/DESRES/2023.2}"

# Step 1: system building
# write msj files
cat >"$sys_build_msj" <<EOF
task {
  task = "desmond:auto"
}

build_geometry {
  add_counterion = {
     ion = Na
     number = neutralize_system
  }
  box = {
     shape = orthorhombic
     size = [12.0 12.0 12.0 ]
     size_type = buffer
  }
  override_forcefield = OPLS_2005
  rezero_system = false
  salt = {
     concentration = 0.15
     negative_ion = Cl
     positive_ion = Na
  }
  solvent = SPC
}

assign_forcefield {
  forcefield = OPLS_2005
}
EOF

# run system building
"${Desmond}/utilities/multisim" -JOBNAME md_setup -maxjob 1 -m "$sys_build_msj" "$in_mae_complex" -o "$sys_build_out" -WAIT

cd "$workdir" || exit
# Step 2: md simulation
# write msj file
cat >"$md_msj" <<EOF
# Desmond standard NPT relaxation protocol
# All times are in the unit of ps.
# Energy is in the unit of kcal/mol.
task {
   task = "desmond:auto"
   set_family = {
      desmond = {
         checkpt.write_last_step = no
      }
   }
}

simulate {
   title       = "Brownian Dynamics NVT, T = 10 K, small timesteps, and restraints on solute heavy atoms, 100ps"
   annealing   = off
   time        = 100
   timestep    = [0.001 0.001 0.003 ]
   temperature = 10.0
   ensemble = {
      class = "NVT"
      method = "Brownie"
      brownie = {
         delta_max = 0.1
      }
   }
   restrain = {
      atom = "solute_heavy_atom"
      force_constant = 50.0
   }
}

simulate {
   title       = "NVT, T = 10 K, small timesteps, and restraints on solute heavy atoms, 12ps"
   annealing   = off
   time        = 12
   timestep    = [0.001 0.001 0.003]
   temperature = 10.0
   restrain    = { atom = solute_heavy_atom force_constant = 50.0 }
   ensemble    = {
      class  = NVT
      method = Langevin
      thermostat.tau = 0.1
   }

   randomize_velocity.interval = 1.0
   eneseq.interval             = 0.3
   trajectory.center           = []
}

simulate {
   title       = "NPT, T = 10 K, and restraints on solute heavy atoms, 12ps"
   annealing   = off
   time        = 12
   temperature = 10.0
   restrain    = retain
   ensemble    = {
      class  = NPT
      method = Langevin
      thermostat.tau = 0.1
      barostat  .tau = 50.0
   }

   randomize_velocity.interval = 1.0
   eneseq.interval             = 0.3
   trajectory.center           = []
}

simulate {
   title       = "NPT and restraints on solute heavy atoms, 12ps"
   effect_if   = [["@*.*.annealing"] 'annealing = off temperature = "@*.*.temperature[0][0]"']
   time        = 12
   restrain    = retain
   ensemble    = {
      class  = NPT
      method = Langevin
      thermostat.tau = 0.1
      barostat  .tau = 50.0
   }

   randomize_velocity.interval = 1.0
   eneseq.interval             = 0.3
   trajectory.center           = []
}

simulate {
   title       = "NPT and no restraints, 24ps"
   effect_if   = [["@*.*.annealing"] 'annealing = off temperature = "@*.*.temperature[0][0]"']
   time        = 24
   ensemble    = {
      class  = NPT
      method = Langevin
      thermostat.tau = 0.1
      barostat  .tau = 2.0
   }

   eneseq.interval   = 0.3
   trajectory.center = solute
}

simulate {
   cfg_file = "$md_cfg"
   jobname  = "\$MASTERJOBNAME"
   dir      = "."
   compress = ""
}

pl_analysis {
    ligand_asl = ""
    protein_asl = ""
}

EOF

# write cfg file
cat >"$md_cfg" <<EOF
annealing = false
backend = {
}
bigger_rclone = false
box = ?
checkpt = {
   first = 0.0
   interval = 240.06
   name = "\$JOBNAME.cpt"
   write_last_step = true
}
cpu = 1
cutoff_radius = 9.0
elapsed_time = 0.0
energy_group = false
eneseq = {
   first = 0.0
   interval = 1.2
   name = "\$JOBNAME\$[_replica\$REPLICA\$].ene"
}
ensemble = {
   barostat = {
      tau = 2.0
   }
   class = NPT
   method = MTK
   thermostat = {
      tau = 1.0
   }
}
glue = solute
maeff_output = {
   center_atoms = solute
   first = 0.0
   interval = 120.0
   name = "\$JOBNAME\$[_replica\$REPLICA\$]-out.cms"
   periodicfix = true
   trjdir = "\$JOBNAME\$[_replica\$REPLICA\$]_trj"
}
meta = false
meta_file = ?
pressure = [1.01325 isotropic ]
randomize_velocity = {
   first = 0.0
   interval = inf
   seed = 2007
   temperature = "@*.temperature"
}
restrain = none
simbox = {
   first = 0.0
   interval = 1.2
   name = "\$JOBNAME\$[_replica\$REPLICA\$]_simbox.dat"
}
surface_tension = 0.0
taper = false
temperature = [
   [300.0 0 ]
]
time = ${md_time}000
timestep = [0.002 0.002 0.006 ]
trajectory = {
   center = []
   first = 0.0
   format = dtr
   frames_per_file = 250
   interval = $md_time
   name = "\$JOBNAME\$[_replica\$REPLICA\$]_trj"
   periodicfix = true
   write_last_vel = false
   write_velocity = false
}

EOF

# run md
"${Desmond}/utilities/multisim" \
  -JOBNAME md \
  -maxjob 1 \
  -m "$md_msj" \
  -c "$md_cfg" \
  -SUBHOST localhost \
  -description 'Molecular Dynamics' "$sys_build_out" \
  -mode umbrella \
  -o "$md_out" \
  -lic DESMOND_ACADEMIC:16 \
  -debug \
  -WAIT
