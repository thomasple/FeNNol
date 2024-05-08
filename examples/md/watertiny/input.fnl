device cuda:0 
matmul_precision highest

model_file ../ani2x.fnx

xyz_input{
  file watertiny.xyz
  # whether the first column is the atom index (Tinker format)
  indexed yes
  # whether a comment line is present
  has_comment_line no
}

# cell vectors (format:  ax ay az bx by bz cx cy cz)
cell = 9.3215 0. 0. 0. 9.3215 0. 0. 0. 9.3215
# compute neighborlists using minimum image convention
minimum_image no 
# whether to wrap the atoms inside the first unit cell
wrap_box no
estimate_pressure no

# number of steps to perform
nsteps = 1000000
# timestep of the dynamics
dt[fs] = .5 

nblist_skin 2.

#time between each saved frame
tdump[ps] = 1.
# number of steps between each printing of the energy
nprint = 1000

## set the thermostat
#thermostat  NVE 
thermostat LGV 
#thermostat ADQTB 

## Thermostat parameters
temperature = 300.
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

