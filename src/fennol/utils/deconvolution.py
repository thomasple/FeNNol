import numpy as np
import jax.numpy as jnp
import jax
from functools import partial


def kernel_lorentz(w, w0, gamma):
    sel = np.logical_and(np.abs(w) < 1.0e-10, np.abs(w0) < 1.0e-10)
    w2 = np.where(sel, 1.0, w**2)
    w02 = np.where(sel, 1.0, w0**2)
    return gamma * w2 / (np.pi * (w2 * gamma**2 + (w2 - w02) ** 2))


def kernel_lorentz_pot(w, w0, gamma):
    sel = np.logical_and(jnp.abs(w) < 1.0e-10, np.abs(w0) < 1.0e-10)
    w2 = np.where(sel, 1.0, w**2)
    w02 = np.where(sel, 1.0, w0**2)
    return gamma * w2 / (np.pi * (w02 * gamma**2 + (w2 - w02) ** 2))


def deconvolute_spectrum(
    s_in,
    omega,
    gamma,
    niteration=10,
    kernel=kernel_lorentz,
    trans=False,
    symmetrize=True,
    thr=1.0e-10,
    verbose=False,
    K_D=None,
):
    assert s_in.shape[0] == omega.shape[0], "s_in and omega must have the same length"
    domega = omega[1] - omega[0]
    if symmetrize:
        nom_save = omega.shape[0]
        s_in = np.concatenate((np.flip(s_in[1:], axis=0), s_in), axis=0)
        omega = np.concatenate((-np.flip(omega[1:], axis=0), omega), axis=0)

    if K_D is not None:
        K, D = K_D
        assert K.shape == (omega.shape[0],omega.shape[0]), "K and omega must have the same length"
        assert D.shape == K.shape, "D and K must have the same shape"
    else:
        if verbose:
            print("deconvolution started.")
            print("  computing kernel matrix...")
        nom = omega.shape[0]
        omij0, omij1 = np.meshgrid(omega, omega)
        omij0 = omij0.flatten(order="F")
        omij1 = omij1.flatten(order="F")
        K = kernel(omij0, omij1, gamma).reshape(nom, nom)
        if trans:
            omnorm = np.arange(nom) * domega
            omnorm = np.concatenate((-np.flip(omnorm[1:], axis=0), omnorm), axis=0)
            omnormij0, omnormij1 = np.meshgrid(omega, omnorm)
            omnormij0 = omnormij0.flatten(order="F")
            omnormij1 = omnormij1.flatten(order="F")
            Knorm = kernel(omnormij0, omnormij1, gamma).reshape(nom, 2 * nom - 1)
            K = K / np.sum(Knorm, axis=1)[:, None] / domega
            del Knorm, omnormij0, omnormij1, omnorm
        else:
            K = K / np.sum(K, axis=0)[None, :] / domega

        if verbose:
            print("  kernel matrix computed.")
            print("  computing double convolution matrix...")

        D = (K.T @ K) * domega

        if verbose:
            print("  double convolution matrix computed.")
            print("  starting iterations.")

    s_out = s_in
    h = (K.T @ s_in) * domega
    for i in range(niteration):
        if verbose:
            print(f"  iteration {i+1}/{niteration}")
        den = (s_out[None, :] * D).sum(axis=1) * domega
        s_next = s_out * h / den
        diff = ((s_next - s_out) ** 2).sum() / (s_out**2).sum()
        if verbose:
           print(f"    relative difference: {diff}")
        s_out = s_next
        if diff < thr:
           break

    if verbose:
        print("deconvolution finished.")
    s_rec = (K @ s_out) * domega
    if symmetrize:
        s_rec = s_rec[nom_save - 1 :]
        s_out = s_out[nom_save - 1 :]

    return s_out, s_rec, (K, D)
