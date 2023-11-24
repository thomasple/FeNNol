import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
import math
import optax

from ..utils.atomic_units import AtomicUnits as au  # CM1,THZ,BOHR,MPROT
from ..utils import Counter
from ..utils.deconvolution import deconvolute_spectrum, kernel_lorentz_pot, kernel_lorentz


def get_thermostat(
    thermostat_name,
    dt,
    mass,
    gamma=None,
    kT=None,
    species=None,
    qtb_parameters=None,
    rng_key=None,
    nbeads=None,
):
    state = {}
    postprocess = lambda x: x

    thermostat_name = str(thermostat_name).upper()
    if thermostat_name in ["LGV", "LANGEVIN"]:
        assert rng_key is not None, "rng_key must be provided for QTB thermostat"
        assert kT is not None, "kT must be specified for QTB thermostat"
        assert gamma is not None, "gamma must be specified for QTB thermostat"
        if nbeads is None:
            a1 = math.exp(-gamma * dt)
            a2 = jnp.asarray(((1 - a1 * a1) * kT / mass[:, None]) ** 0.5)
        else:
            if isinstance(gamma, float):
                gamma = np.array([gamma] * nbeads)
            assert isinstance(gamma, np.ndarray), "gamma must be a float or a numpy array"
            assert gamma.shape[0] == nbeads, "gamma must have the same length as nbeads"
            a1 = np.exp(-gamma * dt)[:,None,None]
            a2 = jnp.asarray(((1 - a1 * a1) * kT / mass[None,:, None]) ** 0.5)

        state["rng_key"] = rng_key

        def thermostat(vel, state):
            rng_key, noise_key = jax.random.split(state["rng_key"])
            noise = jax.random.normal(noise_key, vel.shape, dtype=vel.dtype)
            vel = a1 * vel + a2 * noise
            return vel, {**state, "rng_key": rng_key}

    elif thermostat_name in [
        "GD",
        "DESCENT",
        "GRADIENT_DESCENT",
        "MIN",
        "MINIMIZE",
    ]:
        assert nbeads is None, "Gradient descent is not compatible with PIMD"
        a1 = math.exp(-gamma * dt)

        def thermostat(vel, state):
            return a1 * vel, state

    elif thermostat_name in ["NVE", "NONE"]:
        thermostat = lambda x, s: (x, s)
    elif thermostat_name in ["QTB","ADQTB"]:
        assert nbeads is None, "QTB is not compatible with PIMD"
        assert (
            qtb_parameters is not None
        ), "qtb_parameters must be provided for QTB thermostat"
        assert rng_key is not None, "rng_key must be provided for QTB thermostat"
        assert kT is not None, "kT must be specified for QTB thermostat"
        assert gamma is not None, "gamma must be specified for QTB thermostat"
        assert species is not None, "species must be provided for QTB thermostat"
        thermostat, postprocess, state = initialize_qtb(
            qtb_parameters,
            dt=dt,
            mass=mass,
            gamma=gamma,
            kT=kT,
            species=species,
            rng_key=rng_key,
            adaptive = thermostat_name.startswith("AD"),
        )
    else:
        raise ValueError(f"Unknown thermostat {thermostat_name}")

    return thermostat, postprocess, state


def initialize_qtb(qtb_parameters, dt, mass, gamma, kT, species, rng_key,adaptive):
    state = {}
    verbose = qtb_parameters.get("verbose", False)

    mass = jnp.asarray(mass, dtype=jnp.float32)

    nat = species.shape[0]
    # define type indices
    species_set = set(species)
    nspecies = len(species_set)
    idx = {sp: i for i, sp in enumerate(species_set)}
    type_idx = np.array([idx[sp] for sp in species], dtype=np.int32)

    n_of_type = np.zeros(nspecies, dtype=np.int32)
    for i in range(nspecies):
        n_of_type[i] = (type_idx == i).nonzero()[0].shape[0]
    n_of_type = jnp.asarray(n_of_type, dtype=jnp.float32)
    mass_idx = jax.ops.segment_sum(mass, type_idx, nspecies) / n_of_type

    corr_kin = qtb_parameters.get("corr_kin", -1)
    do_corr_kin = corr_kin <= 0
    if do_corr_kin:
        corr_kin = 1.0
    state["corr_kin"] = corr_kin
    state["corr_kin_prev"] = corr_kin
    state["do_corr_kin"] = do_corr_kin
    state["isame_kin"] = 0

    # spectra parameters
    omegasmear = np.pi / dt / 100.0
    Tseg = qtb_parameters.get("tseg", 1.0 / au.PS)
    nseg = int(Tseg / dt)
    Tseg = nseg * dt
    dom = 2 * np.pi / (3 * Tseg)
    omegacut = qtb_parameters.get("omegacut", 15000.0 / au.CM1)
    nom = int(omegacut / dom)
    omega = dom * np.arange((3 * nseg) // 2 + 1)
    cutoff = jnp.asarray(1.0 / (1.0 + np.exp((omega - omegacut) / omegasmear)))
    assert (
        omegacut < omega[-1]
    ), f"omegacut must be smaller than {omega[-1]*au.CM1} CM-1"

    # initialize gammar
    assert (
        gamma < 0.5 * omegacut
    ), "gamma must be much smaller than omegacut (at most 0.5*omegacut)"
    gammar_min = qtb_parameters.get("gammar_min", 0.1)
    state["gammar"] = jnp.asarray(np.ones((nspecies, nom), dtype=np.float32))

    # Ornstein-Uhlenbeck correction for colored noise
    a1 = np.exp(-gamma * dt)
    OUcorr = jnp.asarray(
        (1.0 - 2.0 * a1 * np.cos(omega * dt) + a1**2)
        / (dt**2 * (gamma**2 + omega**2))
    )

    # hbar schedule
    classical_kernel = qtb_parameters.get("classical_kernel", False)
    hbar = qtb_parameters.get("hbar", 1.0)
    u = 0.5 * hbar * np.abs(omega) / kT
    theta = kT * np.ones_like(omega)
    if hbar > 0:
        theta[1:] *= u[1:] / np.tanh(u[1:])
    theta = jnp.asarray(theta, dtype=jnp.float32)

    noise_key, state["rng_key"] = jax.random.split(rng_key)
    del rng_key
    state["white_noise"] = jax.random.normal(
        noise_key, (3 * nseg, nat, 3), dtype=jnp.float32
    )
    state["force"] = jnp.zeros((nseg, nat, 3), dtype=jnp.float32)

    startsave = qtb_parameters.get("startsave", 1)
    counter = Counter(nseg, startsave=startsave)
    state["istep"] = 0
    state["nadapt"] = 0
    state["nsample"] = 0

    write_spectra = qtb_parameters.get("write_spectra", True)
    do_compute_spectra = write_spectra or adaptive

    if do_compute_spectra:
        state["vel"] = jnp.zeros((nseg, nat, 3), dtype=jnp.float32)

        state["dFDT"] = jnp.zeros((nspecies, nom), dtype=jnp.float32)
        state["mCvv"] = jnp.zeros((nspecies, nom), dtype=jnp.float32)
        state["Cvf"] = jnp.zeros((nspecies, nom), dtype=jnp.float32)
        state["Cff"] = jnp.zeros((nspecies, nom), dtype=jnp.float32)
        state["dFDT_avg"] = jnp.zeros((nspecies, nom), dtype=jnp.float32)
        state["mCvv_avg"] = jnp.zeros((nspecies, nom), dtype=jnp.float32)
        state["Cvfg_avg"] = jnp.zeros((nspecies, nom), dtype=jnp.float32)
        state["Cff_avg"] = jnp.zeros((nspecies, nom), dtype=jnp.float32)

    
    # adaptation parameters
    if adaptive:
        skipseg = qtb_parameters.get("skipseg", 1)

        adaptation_method = str(qtb_parameters.get("adaptation_method", "ADABELIEF")).upper().strip()
        authorized_methods = ["SIMPLE","RATIO","ADABELIEF"]
        assert adaptation_method in authorized_methods, f"adaptation_method must be one of {authorized_methods}"
        if adaptation_method == "SIMPLE":
            agamma = qtb_parameters.get("agamma", 1.e-3)
            assert agamma>0, "agamma must be positive"
            a1_ad = agamma  * Tseg #  * gamma
            print(f"ADQTB SIMPLE: agamma = {agamma:.3f}")
            def update_gammar(state):
                g = state["dFDT"]
                gammar = state["gammar"] - a1_ad*g
                gammar = jnp.maximum(gammar_min,gammar)
                return {**state, "gammar": gammar}
        elif adaptation_method == "RATIO":
            tau_ad = qtb_parameters.get("tau_ad", 5./au.PS)
            tau_s = qtb_parameters.get("tau_s", 10*tau_ad)
            assert tau_ad>0, "tau_ad must be positive"
            print(f"ADQTB RATIO: tau_ad = {tau_ad*au.PS:.2f} ps, tau_s = {tau_s*au.PS:.2f} ps")
            b1 = np.exp(-Tseg/tau_ad)
            b2 = np.exp(-Tseg/tau_s)
            state["mCvv_m"] = jnp.zeros((nspecies, nom), dtype=np.float32)
            state["Cvf_m"] = jnp.zeros((nspecies, nom), dtype=np.float32)
            state["n_adabelief"] = 0
            def update_gammar(state):
                n_adabelief = state["n_adabelief"] + 1
                mCvv_m = state["mCvv_m"]*b1 + state["mCvv"]*(1.-b1)
                Cvf_m = state["Cvf_m"]*b2 + state["Cvf"]*(1.-b2)
                mCvv = mCvv_m/(1.-b1**n_adabelief)
                Cvf = Cvf_m/(1.-b2**n_adabelief)
                # g = Cvf/(mCvv+1.e-8)-state["gammar"]
                gammar = Cvf/(mCvv+1.e-8)
                gammar = jnp.maximum(gammar_min,gammar)
                return {**state, "gammar": gammar,"mCvv_m":mCvv_m,"Cvf_m":Cvf_m,"n_adabelief":n_adabelief}
            
        elif adaptation_method == "ADABELIEF":
            agamma = qtb_parameters.get("agamma", 1.e-2)
            tau_ad = qtb_parameters.get("tau_ad", 1./au.PS)
            tau_s = qtb_parameters.get("tau_s", 100*tau_ad)
            assert tau_ad>0, "tau_ad must be positive"
            assert tau_s>0, "tau_s must be positive"
            assert agamma>0, "agamma must be positive"
            print(f"ADQTB ADABELIEF: agamma = {agamma:.3f}, tau_ad = {tau_ad*au.PS:.2f} ps, tau_s = {tau_s*au.PS:.2f} ps")

            a1_ad = agamma #* Tseg #* gamma
            b1 = np.exp(-Tseg/tau_ad)
            b2 = np.exp(-Tseg/tau_s)
            state["dFDT_m"] = jnp.zeros((nspecies, nom), dtype=np.float32)
            state["dFDT_s"] = jnp.zeros((nspecies, nom), dtype=np.float32)
            state["n_adabelief"] = 0
            def update_gammar(state):
                n_adabelief = state["n_adabelief"] + 1
                dFDT = state["dFDT"]
                dFDT_m = state["dFDT_m"]*b1 + dFDT*(1.-b1)
                dFDT_s = state["dFDT_s"]*b2 + (dFDT-dFDT_m)**2*(1.-b2) + 1.e-8
                #bias correction
                mt = dFDT_m/(1.-b1**n_adabelief)
                st = dFDT_s/(1.-b2**n_adabelief)
                gammar = state["gammar"] - a1_ad* mt/(st**0.5 + 1.e-8)
                gammar = jnp.maximum(gammar_min,gammar)
                return {**state, "gammar": gammar,"dFDT_m":dFDT_m,"n_adabelief":n_adabelief,"dFDT_s":dFDT_s} 


    else:
        update_gammar = lambda x: x

    
    
    def compute_corr_pot(niter=20,verbose=False):
        if classical_kernel or hbar==0:
            return np.ones(nom)
        
        s_0=np.array((theta/kT*cutoff)[:nom])
        s_out,s_rec,_=deconvolute_spectrum(s_0,omega[:nom]
                ,gamma,niter,kernel=kernel_lorentz_pot,trans=True
                ,symmetrize=True,verbose=verbose)
        corr_pot=1.+(s_out-s_0)/s_0
        columns=np.column_stack((omega[:nom]*au.CM1
                                    ,corr_pot-1.
                                    ,s_0,s_out,s_rec)
                    )
        np.savetxt('corr_pot.dat',columns,header='omega(cm-1) corr_pot s_0 s_out s_rec')
        return corr_pot
    
    def compute_corr_kin(state,niter=7,verbose=False):
        if not state["do_corr_kin"]:
            return state
        if classical_kernel or hbar==0:
            return 1.
        
        K_D = state.get("K_D",None)
        mCvv=(state["mCvv_avg"][:,:nom]*n_of_type[:,None]).sum(axis=0)/nat
        s_0=np.array(mCvv*kT/theta[:nom]/state["corr_pot"])
        s_out,s_rec,K_D=deconvolute_spectrum(s_0,omega[:nom]
                    ,gamma,niter,kernel=kernel_lorentz,trans=False
                    ,symmetrize=True,verbose=verbose,K_D=K_D)
        s_out=s_out*theta[:nom]/kT
        s_rec=s_rec*theta[:nom]/kT*state["corr_pot"]
        mCvvsum = mCvv.sum()
        rec_ratio=mCvvsum/s_rec.sum()
        if rec_ratio<0.95 or rec_ratio>1.05:
            print("WARNING: reconvolution error is too high, corr_kin was not updated")
            return
        
        corr_kin = mCvvsum/s_out.sum()
        if np.abs(corr_kin - state["corr_kin_prev"]) < 1.e-4:
            isame_kin = state["isame_kin"]+1
        else:
            isame_kin = 0
        
        print("corr_kin: ", corr_kin)
        do_corr_kin = state["do_corr_kin"]
        if isame_kin > 10:
            print("INFO: corr_kin is converged (it did not change for 10 consecutive segments)")
            do_corr_kin = False
        
        return {**state, "corr_kin": corr_kin, "corr_kin_prev": corr_kin, "isame_kin": isame_kin, "do_corr_kin": do_corr_kin,"K_D":K_D}


    @jax.jit
    def ff_kernel(state):
        if classical_kernel:
            kernel = cutoff * (2 * gamma * kT / dt)
        else:
            kernel = theta * cutoff * OUcorr * (2 * gamma / dt)
        gamma_ratio = jnp.concatenate(
            (
                state["gammar"].T * state["corr_pot"][:, None],
                jnp.ones((kernel.shape[0] - nom, nspecies)),
            ),
            axis=0,
        )
        return kernel[:, None] * gamma_ratio * mass_idx[None, :]

    @jax.jit
    def refresh_force(state):
        rng_key, noise_key = jax.random.split(state["rng_key"])
        white_noise = jnp.concatenate(
            (
                state["white_noise"][nseg:],
                jax.random.normal(noise_key, (nseg, nat, 3), dtype=jnp.float32),
            ),
            axis=0,
        )
        amplitude = ff_kernel(state) ** 0.5
        s = jnp.fft.rfft(white_noise, 3 * nseg, axis=0) * amplitude[:, type_idx, None]
        force = jnp.fft.irfft(s, 3 * nseg, axis=0)[nseg : 2 * nseg]
        return {**state, "rng_key": rng_key, "white_noise": white_noise, "force": force}

    @jax.jit
    def compute_spectra(state):
        sf = jnp.fft.rfft(state["force"]/gamma, 3 * nseg, axis=0, norm="ortho")
        sv = jnp.fft.rfft(state["vel"], 3 * nseg, axis=0, norm="ortho")
        Cvv = jnp.sum(jnp.abs(sv[:nom]) ** 2, axis=-1).T
        Cff = jnp.sum(jnp.abs(sf[:nom]) ** 2, axis=-1).T
        Cvf = jnp.sum(jnp.real(sv[:nom] * jnp.conj(sf[:nom])), axis=-1).T

        mCvv = (
            (dt / 3.0)
            * jnp.zeros_like(state["mCvv"]).at[type_idx].add(Cvv)
            * mass_idx[:, None]
            / n_of_type[:, None]
        )
        Cvf = (
            (dt / 3.0)
            * jnp.zeros_like(state["Cvf"]).at[type_idx].add(Cvf)
            / n_of_type[:, None]
        )
        Cff = (
            (dt / 3.0)
            * jnp.zeros_like(state["Cff"]).at[type_idx].add(Cff)
            / n_of_type[:, None]
        )
        dFDT = mCvv * state["gammar"] - Cvf

        nsinv = 1.0 / state["nsample"]
        b1 = 1.0 - nsinv
        dFDT_avg = state["dFDT_avg"] * b1 + dFDT * nsinv
        mCvv_avg = state["mCvv_avg"] * b1 + mCvv * nsinv
        Cvfg_avg = state["Cvfg_avg"] * b1 + Cvf/state["gammar"] * nsinv
        Cff_avg = state["Cff_avg"] * b1 + Cff * nsinv

        return {
            **state,
            "mCvv": mCvv,
            "Cvf": Cvf,
            "Cff": Cff,
            "dFDT": dFDT,
            "dFDT_avg": dFDT_avg,
            "mCvv_avg": mCvv_avg,
            "Cvfg_avg": Cvfg_avg,
            "Cff_avg": Cff_avg,
        }

    def write_spectra_to_file(state):
        mCvv_avg = np.array(state["mCvv_avg"])
        Cvfg_avg = np.array(state["Cvfg_avg"])
        Cff_avg = np.array(state["Cff_avg"])*3./dt/(gamma**2)
        dFDT_avg = np.array(state["dFDT_avg"])
        gammar = np.array(state["gammar"])
        Cff_theo = np.array(ff_kernel(state))[:nom].T
        for i, sp in enumerate(species_set):
            ff_scale = au.KELVIN / ((2 * gamma / dt) * mass_idx[i])
            columns = np.column_stack(
                (
                    omega[:nom] * au.CM1,
                    mCvv_avg[i],
                    Cvfg_avg[i],
                    dFDT_avg[i],
                    gammar[i] *gamma* au.THZ,
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
            print("QTB spectra written.")

    @jax.jit
    def thermostat(vel, state):
        istep = state["istep"]
        dvel = dt * state["force"][istep] / mass[:, None]
        if do_compute_spectra:
            vel2 = state["vel"].at[istep].set(vel * a1**0.5 + 0.5 * dvel)
            return vel * a1 + dvel, {
                **state,
                "vel": vel2,
            }
        else:
            return vel * a1 + dvel, state

    @jax.jit
    def postprocess_work(state):
        if do_compute_spectra:
            state = compute_spectra(state)
        if adaptive:
            state = jax.lax.cond(state["nadapt"]>skipseg,update_gammar,lambda x:x,state)
        state = refresh_force(state)
        return state

    def postprocess(state):
        counter.increment()
        state = {**state, "istep": counter.count}
        if not counter.is_reset_step:
            return state
        state["nadapt"] += 1
        state["nsample"] = max(state["nadapt"] - startsave + 1, 1)
        if verbose:
            print("Refreshing QTB forces.")
        state = postprocess_work(state)
        state = compute_corr_kin(state)
        if write_spectra:
            write_spectra_to_file(state)
        return state

    state["corr_pot"] = jnp.asarray(compute_corr_pot(), dtype=jnp.float32)

    return thermostat, postprocess, refresh_force(state)
