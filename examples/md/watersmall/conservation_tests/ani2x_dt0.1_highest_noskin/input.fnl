device cuda:0 
matmul_prec highest
per_atom_energy no

model_file ../../../ani2x.fnx

xyz_input{
  file ../../watersmall.xyz
  # whether the first column is the atom index (Tinker format)
  indexed yes
  # whether a comment line is present
  has_comment_line no
}

# cell vectors (format:  ax ay az bx by bz cx cy cz)
cell = 18.643 0. 0. 0. 18.643 0. 0. 0. 18.643
# compute neighborlists using minimum image convention
minimum_image yes
# whether to wrap the atoms inside the first unit cell
wrap_box no
estimate_pressure no

# number of steps to perform
nsteps = 10000000
# timestep of the dynamics
dt[fs] = .1 

#nblist_skin 2.

#time between each saved frame
tdump[ps] = 10
# number of steps between each printing of the energy
nprint = 1000

## set the thermostat
thermostat  NVE 
#thermostat LGV 
#thermostat ADQTB 

## Thermostat parameters
temperature = 0.
#friction constant
gamma[THz] = 20.

## parameters for the Quantum Thermal Bath
qtb{
  # segment length for noise generation and spectra
  tseg[ps]=0.25
  # maximum frequency
  omegacut[cm1]=15000.
  # number of segments to skip before adaptation
  skipseg = 5
  # number of segments to wait before accumulation statistics on spectra
  startsave = 50
  # parameter controlling speed of adQTB adaptation 
  agamma  = 1.
}

