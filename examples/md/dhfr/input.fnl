device cuda:0 
#enable_x64
matmul_prec highest
#print_timings yes

model_file ../ani2x.fnx

xyz_input{
  file dhfr2.xyz
  # whether the first column is the atom index (Tinker format)
  indexed yes
  # whether a comment line is present
  has_comment_line no
}

#0 cell vectors (format:  ax ay az bx by bz cx cy cz)
cell = 62.23 0. 0. 0. 62.23 0. 0. 0. 62.23
# compute neighborlists using minimum image convention
minimum_image yes
# whether to wrap the atoms inside the first unit cell
wrap_box no
estimate_pressure no

# number of steps to perform
nsteps = 100000
# timestep of the dynamics
dt[fs] =  .5
traj_format xyz

nblist_skin 2.

#time between each saved frame
tdump[ps] = 10.
# number of steps between each printing of the energy
nprint = 10
nsummary = 100
nblist_verbose

## set the thermostat
#thermostat  NVE 
#thermostat  NOSE 
thermostat LGV 
#thermostat ADQTB 

## Thermostat parameters
temperature = 300.
#friction constant
gamma[THz] = 10.

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

