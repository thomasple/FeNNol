import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
import math
import optax
import os

from ..utils.atomic_units import AtomicUnits as au  # CM1,THZ,BOHR,MPROT
from ..utils import Counter
from ..utils.deconvolution import (
    deconvolute_spectrum,
    kernel_lorentz_pot,
    kernel_lorentz,
)


def get_thermostat(simulation_parameters, dt, system_data, fprec, rng_key=None):
    state = {}
    postprocess = None
    

    thermostat_name = str(simulation_parameters.get("thermostat", "LGV")).upper()
    compute_thermostat_energy = simulation_parameters.get(
        "include_thermostat_energy", False
    )

    kT = system_data.get("kT", None)
    nbeads = system_data.get("nbeads", None)
    mass = system_data["mass"]
    gamma = simulation_parameters.get("gamma", 1.0 / au.THZ) / au.FS
    species = system_data["species"]

    if nbeads is not None:
        trpmd_lambda = simulation_parameters.get("trpmd_lambda", 1.0)
        gamma = np.maximum(trpmd_lambda * system_data["omk"], gamma)

    if thermostat_name in ["LGV", "LANGEVIN", "FFLGV"]:
        assert rng_key is not None, "rng_key must be provided for QTB thermostat"
        assert kT is not None, "kT must be specified for QTB thermostat"
        assert gamma is not None, "gamma must be specified for QTB thermostat"
        rng_key, v_key = jax.random.split(rng_key)
        if nbeads is None:
            a1 = math.exp(-gamma * dt)
            a2 = jnp.asarray(((1 - a1 * a1) * kT / mass[:, None]) ** 0.5, dtype=fprec)
            vel = (
                jax.random.normal(v_key, (mass.shape[0], 3), dtype=fprec)
                * (kT / mass[:, None]) ** 0.5
            )
        else:
            if isinstance(gamma, float):
                gamma = np.array([gamma] * nbeads)
            assert isinstance(
                gamma, np.ndarray
            ), "gamma must be a float or a numpy array"
            assert gamma.shape[0] == nbeads, "gamma must have the same length as nbeads"
            a1 = np.exp(-gamma * dt)[:, None, None]
            a2 = jnp.asarray(
                ((1 - a1 * a1) * kT / mass[None, :, None]) ** 0.5, dtype=fprec
            )
            vel = (
                jax.random.normal(v_key, (nbeads, mass.shape[0], 3), dtype=fprec)
                * (kT / mass[:, None]) ** 0.5
            )

        state["rng_key"] = rng_key
        if compute_thermostat_energy:
            state["thermostat_energy"] = 0.0
        if thermostat_name == "FFLGV":
            def thermostat(vel, state):
                rng_key, noise_key = jax.random.split(state["rng_key"])
                noise = jax.random.normal(noise_key, vel.shape, dtype=vel.dtype)
                norm_vel = jnp.linalg.norm(vel, axis=-1, keepdims=True)
                dirvel = vel / norm_vel
                if compute_thermostat_energy:
                    v2 = (vel**2).sum(axis=-1)
                vel = a1 * vel + a2 * noise
                new_norm_vel = jnp.linalg.norm(vel, axis=-1, keepdims=True)
                vel = dirvel * new_norm_vel
                new_state = {**state, "rng_key": rng_key}
                if compute_thermostat_energy:
                    v2new = (vel**2).sum(axis=-1)
                    new_state["thermostat_energy"] = (
                        state["thermostat_energy"] + 0.5 * (mass * (v2 - v2new)).sum()
                    )

                return vel, new_state

        else:
            def thermostat(vel, state):
                rng_key, noise_key = jax.random.split(state["rng_key"])
                noise = jax.random.normal(noise_key, vel.shape, dtype=vel.dtype)
                if compute_thermostat_energy:
                    v2 = (vel**2).sum(axis=-1)
                vel = a1 * vel + a2 * noise
                new_state = {**state, "rng_key": rng_key}
                if compute_thermostat_energy:
                    v2new = (vel**2).sum(axis=-1)
                    new_state["thermostat_energy"] = (
                        state["thermostat_energy"] + 0.5 * (mass * (v2 - v2new)).sum()
                    )
                return vel, new_state

    elif thermostat_name in ["BUSSI"]:
        assert rng_key is not None, "rng_key must be provided for QTB thermostat"
        assert kT is not None, "kT must be specified for QTB thermostat"
        assert gamma is not None, "gamma must be specified for QTB thermostat"
        assert nbeads is None, "Bussi thermostat is not compatible with PIMD"
        rng_key, v_key = jax.random.split(rng_key)

        a1 = math.exp(-gamma * dt)
        a2 = (1 - a1) * kT
        vel = (
            jax.random.normal(v_key, (mass.shape[0], 3), dtype=fprec)
            * (kT / mass[:, None]) ** 0.5
        )

        state["rng_key"] = rng_key
        if compute_thermostat_energy:
            state["thermostat_energy"] = 0.0

        def thermostat(vel, state):
            rng_key, noise_key = jax.random.split(state["rng_key"])
            new_state = {**state, "rng_key": rng_key}
            noise = jax.random.normal(noise_key, vel.shape, dtype=vel.dtype)
            R2 = jnp.sum(noise**2)
            R1 = noise[0, 0]
            c = a2 / (mass[:, None] * vel**2).sum()
            d = (a1 * c) ** 0.5
            scale = (a1 + c * R2 + 2 * d * R1) ** 0.5
            if compute_thermostat_energy:
                dek = 0.5 * (mass[:, None] * vel**2).sum() * (scale**2 - 1)
                new_state["thermostat_energy"] = state["thermostat_energy"] + dek
            return scale * vel, new_state

    elif thermostat_name in [
        "GD",
        "DESCENT",
        "GRADIENT_DESCENT",
        "MIN",
        "MINIMIZE",
    ]:
        assert nbeads is None, "Gradient descent is not compatible with PIMD"
        a1 = math.exp(-gamma * dt)

        if nbeads is None:
            vel = jnp.zeros((mass.shape[0], 3), dtype=fprec)
        else:
            vel = jnp.zeros((nbeads, mass.shape[0], 3), dtype=fprec)

        def thermostat(vel, state):
            return a1 * vel, state

    elif thermostat_name in ["NVE", "NONE"]:
        if nbeads is None:
            vel = (
                jax.random.normal(rng_key, (mass.shape[0], 3), dtype=fprec)
                * (kT / mass[:, None]) ** 0.5
            )
            kTsys = jnp.sum(mass[:, None] * vel**2) / (mass.shape[0] * 3)
            vel = vel * (kT / kTsys) ** 0.5
        else:
            vel = (
                jax.random.normal(rng_key, (nbeads, mass.shape[0], 3), dtype=fprec)
                * (kT / mass[None, :, None]) ** 0.5
            )
            kTsys = jnp.sum(mass[None, :, None] * vel**2, axis=(1, 2)) / (
                mass.shape[0] * 3
            )
            vel = vel * (kT / kTsys[:, None, None]) ** 0.5
        thermostat = lambda x, s: (x, s)

    elif thermostat_name in ["NOSE", "NOSEHOOVER", "NOSE_HOOVER"]:
        assert gamma is not None, "gamma must be specified for QTB thermostat"
        ndof = mass.shape[0] * 3
        nkT = ndof * kT
        nose_mass = nkT / gamma**2
        assert nbeads is None, "Nose-Hoover is not compatible with PIMD"
        state["nose_s"] = 0.0
        state["nose_v"] = 0.0
        if compute_thermostat_energy:
            state["thermostat_energy"] = 0.0
        print(
            "# WARNING: Nose-Hoover thermostat is not well tested yet. Energy conservation is not guaranteed."
        )
        vel = (
            jax.random.normal(rng_key, (mass.shape[0], 3), dtype=fprec)
            * (kT / mass[:, None]) ** 0.5
        )

        def thermostat(vel, state):
            nose_s = state["nose_s"]
            nose_v = state["nose_v"]
            kTsys = jnp.sum(mass[:, None] * vel**2)
            nose_v = nose_v + (0.5 * dt / nose_mass) * (kTsys - nkT)
            nose_s = nose_s + dt * nose_v
            vel = jnp.exp(-nose_v * dt) * vel
            kTsys = jnp.sum(mass[:, None] * vel**2)
            nose_v = nose_v + (0.5 * dt / nose_mass) * (kTsys - nkT)
            new_state = {**state, "nose_s": nose_s, "nose_v": nose_v}

            if compute_thermostat_energy:
                new_state["thermostat_energy"] = (
                    nkT * nose_s + (0.5 * nose_mass) * nose_v**2
                )
            return vel, new_state

    elif thermostat_name in ["QTB", "ADQTB"]:
        assert nbeads is None, "QTB is not compatible with PIMD"
        qtb_parameters = simulation_parameters.get("qtb", None)
        assert (
            qtb_parameters is not None
        ), "qtb_parameters must be provided for QTB thermostat"
        assert rng_key is not None, "rng_key must be provided for QTB thermostat"
        assert kT is not None, "kT must be specified for QTB thermostat"
        assert gamma is not None, "gamma must be specified for QTB thermostat"
        assert species is not None, "species must be provided for QTB thermostat"
        rng_key, v_key = jax.random.split(rng_key)
        vel = (
            jax.random.normal(v_key, (mass.shape[0], 3), dtype=fprec)
            * (kT / mass[:, None]) ** 0.5
        )

        thermostat, postprocess, qtb_state = initialize_qtb(
            qtb_parameters,
            fprec=fprec,
            dt=dt,
            mass=mass,
            gamma=gamma,
            kT=kT,
            species=species,
            rng_key=rng_key,
            adaptive=thermostat_name.startswith("AD"),
            compute_thermostat_energy=compute_thermostat_energy,
        )
        state = {**state, **qtb_state}

    elif thermostat_name in ["ANNEAL", "ANNEALING"]:
        assert rng_key is not None, "rng_key must be provided for QTB thermostat"
        assert kT is not None, "kT must be specified for QTB thermostat"
        assert gamma is not None, "gamma must be specified for QTB thermostat"
        assert nbeads is None, "ANNEAL is not compatible with PIMD"
        a1 = math.exp(-gamma * dt)
        a2 = jnp.asarray(((1 - a1 * a1) * kT / mass[:, None]) ** 0.5, dtype=fprec)

        anneal_parameters = simulation_parameters.get("annealing", {})
        init_factor = anneal_parameters.get("init_factor", 1.0 / 25.0)
        assert init_factor > 0.0, "init_factor must be positive"
        final_factor = anneal_parameters.get("final_factor", 1.0 / 10000.0)
        assert final_factor > 0.0, "final_factor must be positive"
        nsteps = simulation_parameters.get("nsteps")
        anneal_steps = anneal_parameters.get("anneal_steps", 1.0)
        assert (
            anneal_steps < 1.0 and anneal_steps > 0.0
        ), "warmup_steps must be between 0 and nsteps"
        pct_start = anneal_parameters.get("warmup_steps", 0.3)
        assert (
            pct_start < 1.0 and pct_start > 0.0
        ), "warmup_steps must be between 0 and nsteps"

        anneal_type = anneal_parameters.get("type", "cosine").lower()
        if anneal_type == "linear":
            schedule = optax.linear_onecycle_schedule(
                peak_value=1.0,
                div_factor=1.0 / init_factor,
                final_div_factor=1.0 / final_factor,
                transition_steps=int(anneal_steps * nsteps),
                pct_start=pct_start,
                pct_final=1.0,
            )
        elif anneal_type == "cosine_onecycle":
            schedule = optax.cosine_onecycle_schedule(
                peak_value=1.0,
                div_factor=1.0 / init_factor,
                final_div_factor=1.0 / final_factor,
                transition_steps=int(anneal_steps * nsteps),
                pct_start=pct_start,
            )
        else:
            raise ValueError(f"Unknown anneal_type {anneal_type}")

        state["rng_key"] = rng_key
        state["istep_anneal"] = 0

        rng_key, v_key = jax.random.split(rng_key)
        Tscale = schedule(0)
        print(f"# ANNEAL: initial temperature = {Tscale*kT*au.KELVIN:.3e} K")
        vel = (
            jax.random.normal(v_key, (mass.shape[0], 3), dtype=fprec)
            * (kT * Tscale / mass[:, None]) ** 0.5
        )

        def thermostat(vel, state):
            rng_key, noise_key = jax.random.split(state["rng_key"])
            noise = jax.random.normal(noise_key, vel.shape, dtype=vel.dtype)

            Tscale = schedule(state["istep_anneal"]) ** 0.5
            vel = a1 * vel + a2 * Tscale * noise
            return vel, {
                **state,
                "rng_key": rng_key,
                "istep_anneal": state["istep_anneal"] + 1,
            }

    else:
        raise ValueError(f"Unknown thermostat {thermostat_name}")

    return thermostat, postprocess, state, vel,thermostat_name


def initialize_qtb(
    qtb_parameters,
    fprec,
    dt,
    mass,
    gamma,
    kT,
    species,
    rng_key,
    adaptive,
    compute_thermostat_energy=False,
):
    state = {}
    post_state = {}
    verbose = qtb_parameters.get("verbose", False)
    if compute_thermostat_energy:
        state["thermostat_energy"] = 0.0

    mass = jnp.asarray(mass, dtype=fprec)

    nat = species.shape[0]
    # define type indices
    species_set = set(species)
    nspecies = len(species_set)
    idx = {sp: i for i, sp in enumerate(species_set)}
    type_idx = np.array([idx[sp] for sp in species], dtype=np.int32)

    n_of_type = np.zeros(nspecies, dtype=np.int32)
    for i in range(nspecies):
        n_of_type[i] = (type_idx == i).nonzero()[0].shape[0]
    n_of_type = jnp.asarray(n_of_type, dtype=fprec)
    mass_idx = jax.ops.segment_sum(mass, type_idx, nspecies) / n_of_type

    corr_kin = qtb_parameters.get("corr_kin", -1)
    do_corr_kin = corr_kin <= 0
    if do_corr_kin:
        corr_kin = 1.0
    state["corr_kin"] = corr_kin
    post_state["corr_kin_prev"] = corr_kin
    post_state["do_corr_kin"] = do_corr_kin
    post_state["isame_kin"] = 0

    # spectra parameters
    omegasmear = np.pi / dt / 100.0
    Tseg = qtb_parameters.get("tseg", 1.0 / au.PS) * au.FS
    nseg = int(Tseg / dt)
    Tseg = nseg * dt
    dom = 2 * np.pi / (3 * Tseg)
    omegacut = qtb_parameters.get("omegacut", 15000.0 / au.CM1) / au.FS
    nom = int(omegacut / dom)
    omega = dom * np.arange((3 * nseg) // 2 + 1)
    cutoff = jnp.asarray(
        1.0 / (1.0 + np.exp((omega - omegacut) / omegasmear)), dtype=fprec
    )
    assert (
        omegacut < omega[-1]
    ), f"omegacut must be smaller than {omega[-1]*au.CM1} CM-1"

    # initialize gammar
    assert (
        gamma < 0.5 * omegacut
    ), "gamma must be much smaller than omegacut (at most 0.5*omegacut)"
    gammar_min = qtb_parameters.get("gammar_min", 0.1)
    # post_state["gammar"] = jnp.asarray(np.ones((nspecies, nom)), dtype=fprec)
    gammar = np.ones((nspecies, nom), dtype=float)
    try:
        for i, sp in enumerate(species_set):
            if not os.path.exists(f"QTB_spectra_{sp}.out"): continue
            data = np.loadtxt(f"QTB_spectra_{sp}.out")
            gammar[i] = data[:, 4]/(gamma*au.FS*au.THZ)
            print(f"# Restored gammar for species {sp} from QTB_spectra_{sp}.out")
    except Exception as e:
        print(f"# Could not restore gammar for all species with exception {e}. Starting from scratch.")
        gammar[:,:] = 1.0
    post_state["gammar"] = jnp.asarray(gammar, dtype=fprec)

    # Ornstein-Uhlenbeck correction for colored noise
    a1 = np.exp(-gamma * dt)
    OUcorr = jnp.asarray(
        (1.0 - 2.0 * a1 * np.cos(omega * dt) + a1**2) / (dt**2 * (gamma**2 + omega**2)),
        dtype=fprec,
    )

    # hbar schedule
    classical_kernel = qtb_parameters.get("classical_kernel", False)
    hbar = qtb_parameters.get("hbar", 1.0) * au.FS
    u = 0.5 * hbar * np.abs(omega) / kT
    theta = kT * np.ones_like(omega)
    if hbar > 0:
        theta[1:] *= u[1:] / np.tanh(u[1:])
    theta = jnp.asarray(theta, dtype=fprec)

    noise_key, post_state["rng_key"] = jax.random.split(rng_key)
    del rng_key
    post_state["white_noise"] = jax.random.normal(
        noise_key, (3 * nseg, nat, 3), dtype=jnp.float32
    )

    startsave = qtb_parameters.get("startsave", 1)
    counter = Counter(nseg, startsave=startsave)
    state["istep"] = 0
    post_state["nadapt"] = 0
    post_state["nsample"] = 0

    write_spectra = qtb_parameters.get("write_spectra", True)
    do_compute_spectra = write_spectra or adaptive

    if do_compute_spectra:
        state["vel"] = jnp.zeros((nseg, nat, 3), dtype=fprec)

        post_state["dFDT"] = jnp.zeros((nspecies, nom), dtype=fprec)
        post_state["mCvv"] = jnp.zeros((nspecies, nom), dtype=fprec)
        post_state["Cvf"] = jnp.zeros((nspecies, nom), dtype=fprec)
        post_state["Cff"] = jnp.zeros((nspecies, nom), dtype=fprec)
        post_state["dFDT_avg"] = jnp.zeros((nspecies, nom), dtype=fprec)
        post_state["mCvv_avg"] = jnp.zeros((nspecies, nom), dtype=fprec)
        post_state["Cvfg_avg"] = jnp.zeros((nspecies, nom), dtype=fprec)
        post_state["Cff_avg"] = jnp.zeros((nspecies, nom), dtype=fprec)

    if not adaptive:
        update_gammar = lambda x: x
    else:
        # adaptation parameters
        skipseg = qtb_parameters.get("skipseg", 1)

        adaptation_method = (
            str(qtb_parameters.get("adaptation_method", "ADABELIEF")).upper().strip()
        )
        authorized_methods = ["SIMPLE", "RATIO", "ADABELIEF"]
        assert (
            adaptation_method in authorized_methods
        ), f"adaptation_method must be one of {authorized_methods}"
        if adaptation_method == "SIMPLE":
            agamma = qtb_parameters.get("agamma", 1.0e-3) / au.FS
            assert agamma > 0, "agamma must be positive"
            a1_ad = agamma * Tseg  #  * gamma
            print(f"# ADQTB SIMPLE: agamma = {agamma*au.FS:.3f}")

            def update_gammar(post_state):
                g = post_state["dFDT"]
                gammar = post_state["gammar"] - a1_ad * g
                gammar = jnp.maximum(gammar_min, gammar)
                return {**post_state, "gammar": gammar}

        elif adaptation_method == "RATIO":
            tau_ad = qtb_parameters.get("tau_ad", 5.0 / au.PS) * au.FS
            tau_s = qtb_parameters.get("tau_s", 10 * tau_ad) * au.FS
            assert tau_ad > 0, "tau_ad must be positive"
            print(
                f"# ADQTB RATIO: tau_ad = {tau_ad*1e-3:.2f} ps, tau_s = {tau_s*1e-3:.2f} ps"
            )
            b1 = np.exp(-Tseg / tau_ad)
            b2 = np.exp(-Tseg / tau_s)
            post_state["mCvv_m"] = jnp.zeros((nspecies, nom), dtype=fprec)
            post_state["Cvf_m"] = jnp.zeros((nspecies, nom), dtype=fprec)
            post_state["n_adabelief"] = 0

            def update_gammar(post_state):
                n_adabelief = post_state["n_adabelief"] + 1
                mCvv_m = post_state["mCvv_m"] * b1 + post_state["mCvv"] * (1.0 - b1)
                Cvf_m = post_state["Cvf_m"] * b2 + post_state["Cvf"] * (1.0 - b2)
                mCvv = mCvv_m / (1.0 - b1**n_adabelief)
                Cvf = Cvf_m / (1.0 - b2**n_adabelief)
                # g = Cvf/(mCvv+1.e-8)-post_state["gammar"]
                gammar = Cvf / (mCvv + 1.0e-8)
                gammar = jnp.maximum(gammar_min, gammar)
                return {
                    **post_state,
                    "gammar": gammar,
                    "mCvv_m": mCvv_m,
                    "Cvf_m": Cvf_m,
                    "n_adabelief": n_adabelief,
                }

        elif adaptation_method == "ADABELIEF":
            agamma = qtb_parameters.get("agamma", 0.1)
            tau_ad = qtb_parameters.get("tau_ad", 1.0 / au.PS) * au.FS
            tau_s = qtb_parameters.get("tau_s", 100 * tau_ad) * au.FS
            assert tau_ad > 0, "tau_ad must be positive"
            assert tau_s > 0, "tau_s must be positive"
            assert agamma > 0, "agamma must be positive"
            print(
                f"# ADQTB ADABELIEF: agamma = {agamma:.3f}, tau_ad = {tau_ad*1.e-3:.2f} ps, tau_s = {tau_s*1.e-3:.2f} ps"
            )

            a1_ad = agamma * gamma  # * Tseg #* gamma
            b1 = np.exp(-Tseg / tau_ad)
            b2 = np.exp(-Tseg / tau_s)
            post_state["dFDT_m"] = jnp.zeros((nspecies, nom), dtype=fprec)
            post_state["dFDT_s"] = jnp.zeros((nspecies, nom), dtype=fprec)
            post_state["n_adabelief"] = 0

            def update_gammar(post_state):
                n_adabelief = post_state["n_adabelief"] + 1
                dFDT = post_state["dFDT"]
                dFDT_m = post_state["dFDT_m"] * b1 + dFDT * (1.0 - b1)
                dFDT_s = (
                    post_state["dFDT_s"] * b2
                    + (dFDT - dFDT_m) ** 2 * (1.0 - b2)
                    + 1.0e-8
                )
                # bias correction
                mt = dFDT_m / (1.0 - b1**n_adabelief)
                st = dFDT_s / (1.0 - b2**n_adabelief)
                gammar = post_state["gammar"] - a1_ad * mt / (st**0.5 + 1.0e-8)
                gammar = jnp.maximum(gammar_min, gammar)
                return {
                    **post_state,
                    "gammar": gammar,
                    "dFDT_m": dFDT_m,
                    "n_adabelief": n_adabelief,
                    "dFDT_s": dFDT_s,
                }

    def compute_corr_pot(niter=20, verbose=False):
        if classical_kernel or hbar == 0:
            return np.ones(nom)

        s_0 = np.array((theta / kT * cutoff)[:nom])
        s_out, s_rec, _ = deconvolute_spectrum(
            s_0,
            omega[:nom],
            gamma,
            niter,
            kernel=kernel_lorentz_pot,
            trans=True,
            symmetrize=True,
            verbose=verbose,
        )
        corr_pot = 1.0 + (s_out - s_0) / s_0
        columns = np.column_stack(
            (omega[:nom] * au.CM1, corr_pot - 1.0, s_0, s_out, s_rec)
        )
        np.savetxt(
            "corr_pot.dat", columns, header="omega(cm-1) corr_pot s_0 s_out s_rec"
        )
        return corr_pot

    def compute_corr_kin(post_state, niter=7, verbose=False):
        if not post_state["do_corr_kin"]:
            return post_state["corr_kin_prev"], post_state
        if classical_kernel or hbar == 0:
            return 1.0, post_state

        K_D = post_state.get("K_D", None)
        mCvv = (post_state["mCvv_avg"][:, :nom] * n_of_type[:, None]).sum(axis=0) / nat
        s_0 = np.array(mCvv * kT / theta[:nom] / post_state["corr_pot"])
        s_out, s_rec, K_D = deconvolute_spectrum(
            s_0,
            omega[:nom],
            gamma,
            niter,
            kernel=kernel_lorentz,
            trans=False,
            symmetrize=True,
            verbose=verbose,
            K_D=K_D,
        )
        s_out = s_out * theta[:nom] / kT
        s_rec = s_rec * theta[:nom] / kT * post_state["corr_pot"]
        mCvvsum = mCvv.sum()
        rec_ratio = mCvvsum / s_rec.sum()
        if rec_ratio < 0.95 or rec_ratio > 1.05:
            print(
                f"# WARNING: reconvolution error {rec_ratio} is too high, corr_kin was not updated"
            )
            return post_state["corr_kin_prev"], post_state

        corr_kin = mCvvsum / s_out.sum()
        if np.abs(corr_kin - post_state["corr_kin_prev"]) < 1.0e-4:
            isame_kin = post_state["isame_kin"] + 1
        else:
            isame_kin = 0

        # print("# corr_kin: ", corr_kin)
        do_corr_kin = post_state["do_corr_kin"]
        if isame_kin > 10:
            print(
                "# INFO: corr_kin is converged (it did not change for 10 consecutive segments)"
            )
            do_corr_kin = False

        return corr_kin, {
            **post_state,
            "corr_kin_prev": corr_kin,
            "isame_kin": isame_kin,
            "do_corr_kin": do_corr_kin,
            "K_D": K_D,
        }

    @jax.jit
    def ff_kernel(post_state):
        if classical_kernel:
            kernel = cutoff * (2 * gamma * kT / dt)
        else:
            kernel = theta * cutoff * OUcorr * (2 * gamma / dt)
        gamma_ratio = jnp.concatenate(
            (
                post_state["gammar"].T * post_state["corr_pot"][:, None],
                jnp.ones(
                    (kernel.shape[0] - nom, nspecies), dtype=post_state["gammar"].dtype
                ),
            ),
            axis=0,
        )
        return kernel[:, None] * gamma_ratio * mass_idx[None, :]

    @jax.jit
    def refresh_force(post_state):
        rng_key, noise_key = jax.random.split(post_state["rng_key"])
        white_noise = jnp.concatenate(
            (
                post_state["white_noise"][nseg:],
                jax.random.normal(
                    noise_key, (nseg, nat, 3), dtype=post_state["white_noise"].dtype
                ),
            ),
            axis=0,
        )
        amplitude = ff_kernel(post_state) ** 0.5
        s = jnp.fft.rfft(white_noise, 3 * nseg, axis=0) * amplitude[:, type_idx, None]
        force = jnp.fft.irfft(s, 3 * nseg, axis=0)[nseg : 2 * nseg]
        return force, {**post_state, "rng_key": rng_key, "white_noise": white_noise}

    @jax.jit
    def compute_spectra(force, vel, post_state):
        sf = jnp.fft.rfft(force / gamma, 3 * nseg, axis=0, norm="ortho")
        sv = jnp.fft.rfft(vel, 3 * nseg, axis=0, norm="ortho")
        Cvv = jnp.sum(jnp.abs(sv[:nom]) ** 2, axis=-1).T
        Cff = jnp.sum(jnp.abs(sf[:nom]) ** 2, axis=-1).T
        Cvf = jnp.sum(jnp.real(sv[:nom] * jnp.conj(sf[:nom])), axis=-1).T

        mCvv = (
            (dt / 3.0)
            * jnp.zeros_like(post_state["mCvv"]).at[type_idx].add(Cvv)
            * mass_idx[:, None]
            / n_of_type[:, None]
        )
        Cvf = (
            (dt / 3.0)
            * jnp.zeros_like(post_state["Cvf"]).at[type_idx].add(Cvf)
            / n_of_type[:, None]
        )
        Cff = (
            (dt / 3.0)
            * jnp.zeros_like(post_state["Cff"]).at[type_idx].add(Cff)
            / n_of_type[:, None]
        )
        dFDT = mCvv * post_state["gammar"] - Cvf

        nsinv = 1.0 / post_state["nsample"]
        b1 = 1.0 - nsinv
        dFDT_avg = post_state["dFDT_avg"] * b1 + dFDT * nsinv
        mCvv_avg = post_state["mCvv_avg"] * b1 + mCvv * nsinv
        Cvfg_avg = post_state["Cvfg_avg"] * b1 + Cvf / post_state["gammar"] * nsinv
        Cff_avg = post_state["Cff_avg"] * b1 + Cff * nsinv

        return {
            **post_state,
            "mCvv": mCvv,
            "Cvf": Cvf,
            "Cff": Cff,
            "dFDT": dFDT,
            "dFDT_avg": dFDT_avg,
            "mCvv_avg": mCvv_avg,
            "Cvfg_avg": Cvfg_avg,
            "Cff_avg": Cff_avg,
        }

    def write_spectra_to_file(post_state):
        mCvv_avg = np.array(post_state["mCvv_avg"])
        Cvfg_avg = np.array(post_state["Cvfg_avg"])
        Cff_avg = np.array(post_state["Cff_avg"]) * 3.0 / dt / (gamma**2)
        dFDT_avg = np.array(post_state["dFDT_avg"])
        gammar = np.array(post_state["gammar"])
        Cff_theo = np.array(ff_kernel(post_state))[:nom].T
        for i, sp in enumerate(species_set):
            ff_scale = au.KELVIN / ((2 * gamma / dt) * mass_idx[i])
            columns = np.column_stack(
                (
                    omega[:nom] * (au.FS * au.CM1),
                    mCvv_avg[i],
                    Cvfg_avg[i],
                    dFDT_avg[i],
                    gammar[i] * gamma * (au.FS * au.THZ),
                    Cff_avg[i] * ff_scale,
                    Cff_theo[i] * ff_scale,
                )
            )
            np.savetxt(
                f"QTB_spectra_{sp}.out",
                columns,
                fmt="%12.6f",
                header="#omega mCvv Cvf dFDT gammar Cff",
            )
        if verbose:
            print("# QTB spectra written.")

    if compute_thermostat_energy:
        state["qtb_energy_flux"] = 0.0

    @jax.jit
    def thermostat(vel, state):
        istep = state["istep"]
        dvel = dt * state["force"][istep] / mass[:, None]
        new_vel = vel * a1 + dvel
        new_state = {**state, "istep": istep + 1}
        if do_compute_spectra:
            vel2 = state["vel"].at[istep].set(vel * a1**0.5 + 0.5 * dvel)
            new_state["vel"] = vel2
        if compute_thermostat_energy:
            dek = 0.5 * (mass[:, None] * (vel**2 - new_vel**2)).sum()
            ekcorr = (
                0.5
                * (mass[:, None] * new_vel**2).sum()
                * (1.0 - 1.0 / state.get("corr_kin", 1.0))
            )
            new_state["qtb_energy_flux"] = state["qtb_energy_flux"] + dek
            new_state["thermostat_energy"] = new_state["qtb_energy_flux"] + ekcorr
        return new_vel, new_state

    @jax.jit
    def postprocess_work(state, post_state):
        if do_compute_spectra:
            post_state = compute_spectra(state["force"], state["vel"], post_state)
        if adaptive:
            post_state = jax.lax.cond(
                post_state["nadapt"] > skipseg, update_gammar, lambda x: x, post_state
            )
        new_force, post_state = refresh_force(post_state)
        return {**state, "force": new_force}, post_state

    def postprocess(state, post_state):
        counter.increment()
        if not counter.is_reset_step:
            return state, post_state
        post_state["nadapt"] += 1
        post_state["nsample"] = max(post_state["nadapt"] - startsave + 1, 1)
        if verbose:
            print("# Refreshing QTB forces.")
        state, post_state = postprocess_work(state, post_state)
        state["corr_kin"], post_state = compute_corr_kin(post_state)
        state["istep"] = 0
        if write_spectra:
            write_spectra_to_file(post_state)
        return state, post_state

    post_state["corr_pot"] = jnp.asarray(compute_corr_pot(), dtype=fprec)

    state["force"], post_state = refresh_force(post_state)
    return thermostat, (postprocess, post_state), state
