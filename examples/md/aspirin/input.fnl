device cuda:0
model_file model_aspirin.fnx

xyz_input{
  file aspirin.xyz 
  # whether the first column is the atom index (Tinker format)
  indexed no 
  # whether a comment line is present
  box_info yes
}

# number of steps to perform
nsteps = 1000000
# timestep of the dynamics
dt[fs] = 0.5 

#time between each saved frame
tdump[ps] = 1
# number of steps between each printing of the energy
nprint = 100

## set the thermostat
#thermostat NVE
thermostat LGV 

## Thermostat parameters
temperature = 300.
#friction constant
gamma[THz] = 1.
