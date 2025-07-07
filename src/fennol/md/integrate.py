import time
import math
import os

import numpy as np
import jax
import jax.numpy as jnp

from ..utils.atomic_units import AtomicUnits as au
from .thermostats import get_thermostat
from .barostats import get_barostat
from .colvars import setup_colvars
from .spectra import initialize_ir_spectrum

from .utils import load_dynamics_restart, get_restart_file,optimize_fire2
from .initial import load_model, load_system_data, initialize_preprocessing


def initialize_dynamics(simulation_parameters, fprec, rng_key):
    ### LOAD MODEL
    model = load_model(simulation_parameters)
    model_energy_unit = au.get_multiplier(model.energy_unit)

    ### Get the coordinates and species from the xyz file
    system_data, conformation = load_system_data(simulation_parameters, fprec)
    system_data["model_energy_unit"] = model_energy_unit
    system_data["model_energy_unit_str"] = model.energy_unit

    ### FINISH BUILDING conformation
    do_restart = os.path.exists(get_restart_file(system_data))
    if do_restart:
        ### RESTART FROM PREVIOUS DYNAMICS
        restart_data = load_dynamics_restart(system_data)
        print("# RESTARTING FROM PREVIOUS DYNAMICS")
        model.preproc_state = restart_data["preproc_state"]
        conformation["coordinates"] = restart_data["coordinates"]
    else:
        restart_data = {}

    ### INITIALIZE PREPROCESSING
    preproc_state, conformation = initialize_preprocessing(
        simulation_parameters, model, conformation, system_data
    )

    minimize = simulation_parameters.get("xyz_input/minimize", False)
    if minimize and not do_restart:
        assert system_data["nreplicas"] == 1, "Minimization is only supported for single replica systems"
        model.preproc_state = preproc_state
        convert = au.KCALPERMOL / model.Ha_to_model_energy
        nat = system_data["nat"]
        def energy_force_fn(coordinates):
            inputs = {**conformation, "coordinates": coordinates}
            e, f, _ = model.energy_and_forces(
                **inputs, gpu_preprocessing=True
            )
            e = float(e[0]) * convert / nat
            f = np.array(f) * convert
            return e, f
        tol = simulation_parameters.get("xyz_input/minimize_ftol", 1e-1*au.BOHR/au.KCALPERMOL)*au.KCALPERMOL/au.BOHR
        print(f"# Minimizing initial configuration with RMS force tolerance = {tol:.1e} kcal/mol/A")
        conformation["coordinates"], success = optimize_fire2(
            conformation["coordinates"],
            energy_force_fn,
            atol=tol,
            max_disp=0.02,
        )
        if success:
            print("# Minimization successful")
        else:
            print("# Warning: Minimization failed, continuing with last configuration")
        # write the minimized coordinates as an xyz file
        from ..utils.io import write_xyz_frame
        with open(system_data["name"]+".opt.xyz", "w") as f:
            write_xyz_frame(f, system_data["symbols"],np.array(conformation["coordinates"]),cell=conformation.get("cells", None))
        print("# Minimized configuration written to", system_data["name"]+".opt.xyz")
        preproc_state = model.preproc_state
        conformation = model.preprocessing.process(preproc_state, conformation)
        system_data["initial_coordinates"] = np.array(conformation["coordinates"]).copy()

    ### get dynamics parameters
    dt = simulation_parameters.get("dt") * au.FS
    dt2 = 0.5 * dt
    mass = system_data["mass"]
    densmass = system_data["totmass_Da"]*(au.MPROT*au.GCM3)
    nat = system_data["nat"]
    dtm = jnp.asarray(dt / mass[:, None], dtype=fprec)
    ek_avg = 0.5 * nat * system_data["kT"] * np.eye(3)

    nreplicas = system_data.get("nreplicas", 1)
    nbeads = system_data.get("nbeads", None)
    if nbeads is not None:
        nreplicas = nbeads
        dtm = dtm[None, :, :]

    ### INITIALIZE DYNAMICS STATE
    system = {"coordinates": conformation["coordinates"]}
    dyn_state = {
        "istep": 0,
        "dt": dt,
        "pimd": nbeads is not None,
        "preproc_state": preproc_state,
        "start_time_ps": restart_data.get("simulation_time_ps", 0.),
    }
    gradient_keys = ["coordinates"]
    thermo_updates = []

    ### INITIALIZE THERMOSTAT
    thermostat_rng, rng_key = jax.random.split(rng_key)
    (
        thermostat,
        thermostat_post,
        thermostat_state,
        initial_vel,
        dyn_state["thermostat_name"],
    ) = get_thermostat(simulation_parameters, dt, system_data, fprec, thermostat_rng,restart_data)
    do_thermostat_post = thermostat_post is not None
    if do_thermostat_post:
        thermostat_post, post_state = thermostat_post
        dyn_state["thermostat_post_state"] = post_state

    system["thermostat"] = thermostat_state
    system["vel"] = restart_data.get("vel", initial_vel).astype(fprec)

    ### PBC
    pbc_data = system_data.get("pbc", None)
    if pbc_data is not None:
        ### INITIALIZE BAROSTAT
        barostat_key, rng_key = jax.random.split(rng_key)
        thermo_update_ensemble, variable_cell, barostat_state = get_barostat(
            thermostat, simulation_parameters, dt, system_data, fprec, barostat_key,restart_data
        )
        estimate_pressure = variable_cell or pbc_data["estimate_pressure"]
        system["barostat"] = barostat_state
        system["cell"] = conformation["cells"][0]
        if estimate_pressure:
            pressure_o_weight = simulation_parameters.get("pressure_o_weight", 0.0)
            assert (
                0.0 <= pressure_o_weight <= 1.0
            ), "pressure_o_weight must be between 0 and 1"
            gradient_keys.append("strain")
        print("# Estimate pressure: ", estimate_pressure)
    else:
        estimate_pressure = False
        variable_cell = False

        def thermo_update_ensemble(x, v, system):
            v, thermostat_state = thermostat(v, system["thermostat"])
            return x, v, {**system, "thermostat": thermostat_state}

    dyn_state["estimate_pressure"] = estimate_pressure
    dyn_state["variable_cell"] = variable_cell
    thermo_updates.append(thermo_update_ensemble)

    if estimate_pressure:
        use_average_Pkin = simulation_parameters.get("use_average_Pkin", False)
        is_qtb = dyn_state["thermostat_name"].endswith("QTB")
        if is_qtb and use_average_Pkin:
            raise ValueError(
                "use_average_Pkin is not compatible with QTB thermostat, please set use_average_Pkin to False"
            )


    ### ENERGY ENSEMBLE
    ensemble_key = simulation_parameters.get("etot_ensemble_key", None)

    ### COLVARS
    colvars_definitions = simulation_parameters.get("colvars", None)
    use_colvars = colvars_definitions is not None
    if use_colvars:
        colvars_calculators, colvars_names = setup_colvars(colvars_definitions)
        dyn_state["colvars"] = colvars_names

    ### IR SPECTRUM
    do_ir_spectrum = simulation_parameters.get("ir_spectrum", False)
    assert isinstance(do_ir_spectrum, bool), "ir_spectrum must be a boolean"
    if do_ir_spectrum:
        is_qtb = dyn_state["thermostat_name"].endswith("QTB")
        model_ir, ir_state, save_dipole, ir_post = initialize_ir_spectrum(
            simulation_parameters, system_data, fprec, dt, is_qtb
        )
        dyn_state["ir_spectrum"] = ir_state

    ### BUILD GRADIENT FUNCTION
    energy_and_gradient = model.get_gradient_function(
        *gradient_keys, jit=True, variables_as_input=True
    )

    ### COLLECT THERMO UPDATES
    if len(thermo_updates) == 1:
        thermo_update = thermo_updates[0]
    else:

        def thermo_update(x, v, system):
            for update in thermo_updates:
                x, v, system = update(x, v, system)
            return x, v, system

    ### RING POLYMER INITIALIZATION
    if nbeads is not None:
        cay_correction = simulation_parameters.get("cay_correction", True)
        omk = system_data["omk"]
        eigmat = system_data["eigmat"]
        cayfact = 1.0 / (4.0 + (dt * omk[1:, None, None]) ** 2) ** 0.5
        if cay_correction:
            axx = jnp.asarray(2 * cayfact)
            axv = jnp.asarray(dt * cayfact)
            avx = jnp.asarray(-dt * cayfact * omk[1:, None, None] ** 2)
        else:
            axx = jnp.asarray(np.cos(omk[1:, None, None] * dt2))
            axv = jnp.asarray(np.sin(omk[1:, None, None] * dt2) / omk[1:, None, None])
            avx = jnp.asarray(-omk[1:, None, None] * np.sin(omk[1:, None, None] * dt2))

        coordinates = conformation["coordinates"].reshape(nbeads, -1, 3)
        eigx = jnp.zeros_like(coordinates).at[0].set(coordinates[0])
        system["coordinates"] = eigx

    ###############################################
    ### DEFINE UPDATE FUNCTION
    @jax.jit
    def update_conformation(conformation, system):
        x = system["coordinates"]
        if nbeads is not None:
            x = jnp.einsum("in,n...->i...", eigmat, x).reshape(nbeads * nat, 3) * (
                nbeads**0.5
            )
        conformation = {**conformation, "coordinates": x}
        if variable_cell:
            conformation["cells"] = system["cell"][None, :, :].repeat(nreplicas, axis=0)

        

        return conformation

    ###############################################
    ### DEFINE INTEGRATION FUNCTIONS
    def integrate_A_half(x0, v0):
        if nbeads is None:
            return x0 + dt2 * v0, v0

        # update coordinates and velocities of a free ring polymer for a half time step
        eigx_c = x0[0] + dt2 * v0[0]
        eigv_c = v0[0]
        eigx = x0[1:] * axx + v0[1:] * axv
        eigv = x0[1:] * avx + v0[1:] * axx

        return (
            jnp.concatenate((eigx_c[None], eigx), axis=0),
            jnp.concatenate((eigv_c[None], eigv), axis=0),
        )

    @jax.jit
    def integrate(system):
        x = system["coordinates"]
        v = system["vel"] + dtm * system["forces"]
        x, v = integrate_A_half(x, v)
        x, v, system = thermo_update(x, v, system)
        x, v = integrate_A_half(x, v)

        return {**system, "coordinates": x, "vel": v}

    ###############################################
    ### DEFINE OBSERVABLE FUNCTION
    @jax.jit
    def update_observables(system, conformation):
        ### POTENTIAL ENERGY AND FORCES
        epot, de, out = energy_and_gradient(model.variables, conformation)
        out["forces"] = -de["coordinates"]
        epot = epot / model_energy_unit
        de = {k: v / model_energy_unit for k, v in de.items()}
        forces = -de["coordinates"]

        if nbeads is not None:
            ### PROJECT FORCES ONTO POLYMER NORMAL MODES
            forces = jnp.einsum(
                "in,i...->n...", eigmat, forces.reshape(nbeads, nat, 3)
            ) * (1.0 / nbeads**0.5)

        system = {
            **system,
            "epot": jnp.mean(epot),
            "forces": forces,
            "energy_gradients": de,
        }

        ### KINETIC ENERGY
        v = system["vel"]
        if nbeads is None:
            corr_kin = system["thermostat"].get("corr_kin", 1.0)
            # ek = 0.5 * jnp.sum(mass[:, None] * v**2) / state_th.get("corr_kin", 1.0)
            ek = (0.5 / nreplicas / corr_kin) * jnp.sum(
                mass[:, None, None] * v[:, :, None] * v[:, None, :], axis=0
            )
        else:
            ek_c = 0.5 * jnp.sum(
                mass[:, None, None] * v[0, :, :, None] * v[0, :, None, :], axis=0
            )
            ek = ek_c - 0.5 * jnp.sum(
                system["coordinates"][1:, :, :, None] * forces[1:, :, None, :],
                axis=(0, 1),
            )
            system["ek_c"] = jnp.trace(ek_c)

        system["ek"] = jnp.trace(ek)
        system["ek_tensor"] = ek

        if estimate_pressure:
            if use_average_Pkin:
                ek = ek_avg
            elif pressure_o_weight != 1.0:
                v = system["vel"] + 0.5 * dtm * system["forces"]
                if nbeads is None:
                    corr_kin = system["thermostat"].get("corr_kin", 1.0)
                    # ek = 0.5 * jnp.sum(mass[:, None] * v**2) / state_th.get("corr_kin", 1.0)
                    ek = (0.5 / nreplicas / corr_kin) * jnp.sum(
                        mass[:, None, None] * v[:, :, None] * v[:, None, :], axis=0
                    )
                else:
                    ek_c = 0.5 * jnp.sum(
                        mass[:, None, None] * v[0, :, :, None] * v[0, :, None, :],
                        axis=0,
                    )
                    ek = ek_c - 0.5 * jnp.sum(
                        system["coordinates"][1:, :, :, None] * forces[1:, :, None, :],
                        axis=(0, 1),
                    )
                b = pressure_o_weight
                ek = (1.0 - b) * ek + b * system["ek_tensor"]

            vir = jnp.mean(de["strain"], axis=0)
            system["virial"] = vir
            out["virial_tensor"] = vir * model_energy_unit
            
            pV =  2 * ek  - vir
            system["PV_tensor"] = pV
            volume = jnp.abs(jnp.linalg.det(system["cell"]))
            Pres = pV / volume
            system["pressure_tensor"] = Pres
            system["pressure"] = jnp.trace(Pres) * (1.0 / 3.0)
            if variable_cell:
                density = densmass / volume
                system["density"] = density
                system["volume"] = volume

        if ensemble_key is not None:
            kT = system_data["kT"]
            dE = (
                jnp.mean(out[ensemble_key], axis=0) / model_energy_unit - system["epot"]
            )
            system["ensemble_weights"] = -dE / kT

        if "total_dipole" in out:
            if nbeads is None:
                system["total_dipole"] = out["total_dipole"][0]
            else:
                system["total_dipole"] = jnp.mean(out["total_dipole"], axis=0)

        if use_colvars:
            coords = system["coordinates"].reshape(-1, nat, 3)[0]
            colvars = {}
            for colvar_name, colvar_calc in colvars_calculators.items():
                colvars[colvar_name] = colvar_calc(coords)
            system["colvars"] = colvars

        return system, out

    ###############################################
    ### IR SPECTRUM
    if do_ir_spectrum:
        # @jax.jit
        # def update_dipole(ir_state,system,conformation):
        #     def mumodel(coords):
        #         out = model_ir._apply(model_ir.variables,{**conformation,"coordinates":coords})
        #         if nbeads is None:
        #             return out["total_dipole"][0]
        #         return out["total_dipole"].sum(axis=0)
        #     dmudqmodel = jax.jacobian(mumodel)

        #     dmudq = dmudqmodel(conformation["coordinates"])
        #     # print(dmudq.shape)
        #     if nbeads is None:
        #         vel = system["vel"].reshape(-1,1,nat,3)[0]
        #         mudot = (vel*dmudq).sum(axis=(1,2))
        #     else:
        #         dmudq = dmudq.reshape(3,nbeads,nat,3)#.mean(axis=1)
        #         vel = (jnp.einsum("in,n...->i...", eigmat, system["vel"]) *  nbeads**0.5
        #         )
        #         # vel = system["vel"][0].reshape(1,nat,3)
        #         mudot = (vel[None,...]*dmudq).sum(axis=(1,2,3))/nbeads

        #     ir_state = save_dipole(mudot,ir_state)
        #     return ir_state
        @jax.jit
        def update_conformation_ir(conformation, system):
            conformation = {
                **conformation,
                "coordinates": system["coordinates"].reshape(-1, nat, 3)[0],
                "natoms": jnp.asarray([nat]),
                "batch_index": jnp.asarray([0] * nat),
                "species": jnp.asarray(system_data["species"].reshape(-1, nat)[0]),
            }
            if variable_cell:
                conformation["cells"] = system["cell"][None, :, :]
                conformation["reciprocal_cells"] = jnp.linalg.inv(system["cell"])[
                    None, :, :
                ]
            return conformation

        @jax.jit
        def update_dipole(ir_state, system, conformation):
            if model_ir is not None:
                out = model_ir._apply(model_ir.variables, conformation)
                q = out.get("charges", jnp.zeros(nat)).reshape((-1, nat))
                dip = out.get("dipoles", jnp.zeros((nat, 3))).reshape((-1, nat, 3))
            else:
                q = system.get("charges", jnp.zeros(nat)).reshape((-1, nat))
                dip = system.get("dipoles", jnp.zeros((nat, 3))).reshape((-1, nat, 3))
            if nbeads is not None:
                q = jnp.mean(q, axis=0)
                dip = jnp.mean(dip, axis=0)
                vel = system["vel"][0]
                pos = system["coordinates"][0]
            else:
                q = q[0]
                dip = dip[0]
                vel = system["vel"].reshape(-1, nat, 3)[0]
                pos = system["coordinates"].reshape(-1, nat, 3)[0]

            if pbc_data is not None:
                cell_reciprocal = (
                    conformation["cells"][0],
                    conformation["reciprocal_cells"][0],
                )
            else:
                cell_reciprocal = None

            ir_state = save_dipole(
                q, vel, pos, dip.sum(axis=0), cell_reciprocal, ir_state
            )
            return ir_state

    ###############################################
    ### GRAPH UPDATES

    nblist_verbose = simulation_parameters.get("nblist_verbose", False)
    nblist_stride = int(simulation_parameters.get("nblist_stride", -1))
    nblist_warmup_time = simulation_parameters.get("nblist_warmup_time", -1.0) * au.FS
    nblist_warmup = int(nblist_warmup_time / dt) if nblist_warmup_time > 0 else 0
    nblist_skin = simulation_parameters.get("nblist_skin", -1.0)
    if nblist_skin > 0:
        if nblist_stride <= 0:
            ## reference skin parameters at 300K (from Tinker-HP)
            ##   => skin of 2 A gives you 40 fs without complete rebuild
            t_ref = 40.0  # FS
            nblist_skin_ref = 2.0  # A
            nblist_stride = int(math.floor(nblist_skin / nblist_skin_ref * t_ref / dt))
        print(
            f"# nblist_skin: {nblist_skin:.2f} A, nblist_stride: {nblist_stride} steps, nblist_warmup: {nblist_warmup} steps"
        )

    if nblist_skin <= 0:
        nblist_stride = 1

    dyn_state["nblist_countdown"] = 0
    dyn_state["print_skin_activation"] = nblist_warmup > 0

    def update_graphs(istep, dyn_state, system, conformation, force_preprocess=False):
        nblist_countdown = dyn_state["nblist_countdown"]
        if nblist_countdown <= 0 or force_preprocess or (istep < nblist_warmup):
            ### FULL NBLIST REBUILD
            dyn_state["nblist_countdown"] = nblist_stride - 1
            preproc_state = dyn_state["preproc_state"]
            conformation = model.preprocessing.process(
                preproc_state, update_conformation(conformation, system)
            )
            preproc_state, state_up, conformation, overflow = (
                model.preprocessing.check_reallocate(preproc_state, conformation)
            )
            dyn_state["preproc_state"] = preproc_state
            if nblist_verbose and overflow:
                print("step", istep, ", nblist overflow => reallocating nblist")
                print("size updates:", state_up)

            if do_ir_spectrum and model_ir is not None:
                conformation_ir = model_ir.preprocessing.process(
                    dyn_state["preproc_state_ir"],
                    update_conformation_ir(dyn_state["conformation_ir"], system),
                )
                (
                    dyn_state["preproc_state_ir"],
                    _,
                    dyn_state["conformation_ir"],
                    overflow,
                ) = model_ir.preprocessing.check_reallocate(
                    dyn_state["preproc_state_ir"], conformation_ir
                )

        else:
            ### SKIN UPDATE
            if dyn_state["print_skin_activation"]:
                if nblist_verbose:
                    print(
                        "step",
                        istep,
                        ", end of nblist warmup phase => activating skin updates",
                    )
                dyn_state["print_skin_activation"] = False

            dyn_state["nblist_countdown"] = nblist_countdown - 1
            conformation = model.preprocessing.update_skin(
                update_conformation(conformation, system)
            )
            if do_ir_spectrum and model_ir is not None:
                dyn_state["conformation_ir"] = model_ir.preprocessing.update_skin(
                    update_conformation_ir(dyn_state["conformation_ir"], system)
                )

        return conformation, dyn_state

    ################################################
    ### DEFINE STEP FUNCTION
    def step(istep, dyn_state, system, conformation, force_preprocess=False):

        dyn_state = {
            **dyn_state,
            "istep": dyn_state["istep"] + 1,
        }

        ### INTEGRATE EQUATIONS OF MOTION
        system = integrate(system)

        ### UPDATE CONFORMATION AND GRAPHS
        conformation, dyn_state = update_graphs(
            istep, dyn_state, system, conformation, force_preprocess
        )

        ## COMPUTE FORCES AND OBSERVABLES
        system, out = update_observables(system, conformation)

        ## END OF STEP UPDATES
        if do_thermostat_post:
            system["thermostat"], dyn_state["thermostat_post_state"] = thermostat_post(
                system["thermostat"], dyn_state["thermostat_post_state"]
            )
        
        if do_ir_spectrum:
            ir_state = update_dipole(
                dyn_state["ir_spectrum"], system, dyn_state["conformation_ir"]
            )
            dyn_state["ir_spectrum"] = ir_post(ir_state)

        return dyn_state, system, conformation, out

    ###########################################################

    print("# Computing initial energy and forces")

    conformation = update_conformation(conformation, system)
    # initialize IR conformation
    if do_ir_spectrum and model_ir is not None:
        dyn_state["preproc_state_ir"], dyn_state["conformation_ir"] = (
            model_ir.preprocessing(
                model_ir.preproc_state,
                update_conformation_ir(conformation, system),
            )
        )

    system, _ = update_observables(system, conformation)

    return step, update_conformation, system_data, dyn_state, conformation, system
