import numpy as np
import jax
import math
import jax.numpy as jnp
import sympy
from sympy.printing.pycode import pycode
from sympy.physics.wigner import clebsch_gordan
from functools import partial


def CG_SU2(j1: int, j2: int, j3: int) -> np.array:
    r"""Clebsch-Gordan coefficients for the direct product of two irreducible representations of :math:`SU(2)`
    Returns
    -------
    `np.Array`
        tensor :math:`C` of shape :math:`(2j_1+1, 2j_2+1, 2j_3+1)`
    """
    C = np.zeros((2 * j1 + 1, 2 * j2 + 1, 2 * j3 + 1))
    for m1 in range(-j1, j1 + 1):
        for m2 in range(-j2, j2 + 1):
            for m3 in range(-j3, j3 + 1):
                C[m1 + j1, m2 + j2, m3 + j3] = float(
                    clebsch_gordan(j1, j2, j3, m1, m2, m3)
                )
    return C


def change_basis_real_to_complex(l: int) -> np.array:
    r"""Change of basis matrix from real to complex spherical harmonics
    https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    adapted from e3nn.o3._wigner
    """
    q = np.zeros((2 * l + 1, 2 * l + 1), dtype=complex)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / 2**0.5
        q[l + m, l - abs(m)] = -1j / 2**0.5
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / 2**0.5
        q[l + m, l - abs(m)] = 1j * (-1) ** m / 2**0.5

    # factor of (-i)**l to make the Clebsch-Gordan coefficients real
    return q * (-1j) ** l


def CG_SO3(j1: int, j2: int, j3: int) -> np.array:
    r"""Clebsch-Gordan coefficients for the direct product of two irreducible representations of :math:`SO(3)`
    Returns
    -------
    `np.array`
        tensor :math:`C` of shape :math:`(2l_1+1, 2l_2+1, 2l_3+1)`
    """
    C = CG_SU2(j1, j2, j3)
    Q1 = change_basis_real_to_complex(j1)
    Q2 = change_basis_real_to_complex(j2)
    Q3 = change_basis_real_to_complex(j3)
    C = np.real(np.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, np.conj(Q3.T), C))
    return C / np.linalg.norm(C)


def generate_spherical_harmonics(
    lmax, normalize=False, print_code=False, jit=False, vmapped=False
):  # pragma: no cover
    r"""returns a function that computes spherical harmonic up to lmax
    (adapted from e3nn)
    """

    def to_frac(x: float):
        from fractions import Fraction

        s = 1 if x >= 0 else -1
        x = x**2
        x = Fraction(x).limit_denominator()
        x = s * sympy.sqrt(x)
        x = sympy.simplify(x)
        return x

    if vmapped:
        fn_str = "def spherical_harmonics_(x,y,z):\n"
        fn_str += "  sh_0_0 = 1.\n"
    else:
        fn_str = "def spherical_harmonics_(vec):\n"
        if normalize:
            fn_str += "  vec = vec/jnp.linalg.norm(vec,axis=-1,keepdims=True)\n"
        fn_str += "  x,y,z = [jax.lax.index_in_dim(vec, i, axis=-1, keepdims=False) for i in range(3)]\n"
        fn_str += "  sh_0_0 = jnp.ones_like(x)\n"

    x_var, y_var, z_var = sympy.symbols("x y z")
    polynomials = [sympy.sqrt(3) * x_var, sympy.sqrt(3) * y_var, sympy.sqrt(3) * z_var]

    def sub_z1(p, names, polynormz):
        p = p.subs(x_var, 0).subs(y_var, 1).subs(z_var, 0)
        for n, c in zip(names, polynormz):
            p = p.subs(n, c)
        return p

    poly_evalz = [sub_z1(p, [], []) for p in polynomials]

    for l in range(1, lmax + 1):
        sh_variables = sympy.symbols(" ".join(f"sh_{l}_{m}" for m in range(2 * l + 1)))

        for n, p in zip(sh_variables, polynomials):
            fn_str += f"  {n} = {pycode(p.evalf())}\n"

        if l == lmax:
            break

        polynomials = [
            sum(
                to_frac(c.item()) * v * sh
                for cj, v in zip(cij, [x_var, y_var, z_var])
                for c, sh in zip(cj, sh_variables)
            )
            for cij in CG_SO3(l + 1, 1, l)
        ]

        poly_evalz = [sub_z1(p, sh_variables, poly_evalz) for p in polynomials]
        norm = sympy.sqrt(sum(p**2 for p in poly_evalz))
        polynomials = [sympy.sqrt(2 * l + 3) * p / norm for p in polynomials]
        poly_evalz = [sympy.sqrt(2 * l + 3) * p / norm for p in poly_evalz]

        polynomials = [sympy.simplify(p, full=True) for p in polynomials]

    u = ",\n        ".join(
        ", ".join(f"sh_{j}_{m}" for m in range(2 * j + 1)) for j in range(l + 1)
    )
    if vmapped:
        fn_str += f"  return jnp.array([\n        {u}\n    ])\n"
    else:
        fn_str += f"  return jnp.stack([\n        {u}\n    ], axis=-1)\n"

    if print_code:
        print(fn_str)
    exec(fn_str)
    sh = locals()["spherical_harmonics_"]
    if jit:
        sh = jax.jit(sh)
    if not vmapped:
        return sh

    if normalize:

        def spherical_harmonics(vec):
            vec = vec / jnp.linalg.norm(vec, axis=-1, keepdims=True)
            x, y, z = [
                jax.lax.index_in_dim(vec, i, axis=-1, keepdims=False) for i in range(3)
            ]
            return jax.vmap(sh)(x, y, z)

    else:

        def spherical_harmonics(vec):
            x, y, z = [
                jax.lax.index_in_dim(vec, i, axis=-1, keepdims=False) for i in range(3)
            ]
            return jax.vmap(sh)(x, y, z)

    if jit:
        spherical_harmonics = jax.jit(spherical_harmonics)
    return spherical_harmonics


@partial(jax.jit, static_argnums=1)
def spherical_to_cartesian_tensor(Q, lmax):
    q = Q[..., 0]
    if lmax == 0:
        return q[..., None]

    mu = Q[..., 1:4]
    if lmax == 1:
        return jnp.concatenate([q[..., None], mu], axis=-1)

    Q22s = Q[..., 4]
    Q21s = Q[..., 5]
    Q20 = Q[..., 6]
    Q21c = Q[..., 7]
    Q22c = Q[..., 8]
    Tzz = -0.5 * Q20 + (0.5 * 3**0.5) * Q22c
    Txx = -0.5 * Q20 - (0.5 * 3**0.5) * Q22c
    Tyy = Q20
    Txz = 0.5 * (3**0.5) * Q22s
    Tyz = 0.5 * (3**0.5) * Q21c
    Txy = 0.5 * (3**0.5) * Q21s

    if lmax == 2:
        return jnp.concatenate(
            [
                q[..., None],
                mu,
                Txx[..., None],
                Tyy[..., None],
                Tzz[..., None],
                Txy[..., None],
                Txz[..., None],
                Tyz[..., None],
            ],
            axis=-1,
        )
