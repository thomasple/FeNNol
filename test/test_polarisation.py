#!/usr/bin/env python3
"""Test for the polarisation model in FENNOL."""
import jax
import jax.numpy as jnp
from fennol import FENNIX
from jax.random import PRNGKey
from fennol.utils import AtomicUnits as Au

###############################
# Test using Tinker reference #
###############################

modules = {
    'energy': {
        'module_name': 'NEURAL_NET',
        'neurons': [32, 1],
        'input_key': 'coordinates',
    },
    'charges': {
        'module_name': 'CHEMICAL_CONSTANT',
        'value': {
            'O': -0.55824546,
            'H': 0.27912273
        },
    },
    'polarisability': {
    'module_name': 'CHEMICAL_CONSTANT',
        'value': {
            'O': 0.9479157,
            'H': 0.4158204
        },
    },
    'electric_field': {
        'module_name': 'ELECTRIC_FIELD',
    },
    'polarisation': {
        'module_name': 'POLARISATION'
    },
}

model_tinker = FENNIX(
    cutoff=5.0,
    rng_key=PRNGKey(0),
    modules=modules
)

species = jnp.array(
    [
        [8, 1, 1]
    ]
).reshape(-1)

coordinates_tinker = jnp.array(
    [
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0]
        ]
    ]
).reshape(-1, 3)

natoms = jnp.array([3])
batch_index = jnp.array([0, 0, 0])
output_tinker = model_tinker(
        species=species,
        coordinates=coordinates_tinker,
        natoms=natoms,
        batch_index=batch_index
    )


def test_tmu():
    """Test the T@mu output of the polarisation model."""
    t_tinker = jnp.array(
        [
            [1.055, 0.000, 0.000, -0.125, 0.000, 0.000, -0.125, 0.000, 0.000],
            [0.000, 1.055, 0.000, 0.000, -0.125, 0.000, 0.000, 0.250, 0.000],
            [0.000, 0.000, 1.055, 0.000, 0.000, 0.250, 0.000, 0.000, -0.125],
            [-0.125, 0.000, 0.000, 2.405, 0.000, 0.000, -0.044, 0.000, 0.000],
            [0.000, -0.125, 0.000, 0.000, 2.405, 0.000, 0.000, 0.022, -0.066],
            [0.000, 0.000, 0.250, 0.000, 0.000, 2.405, 0.000, -0.066, 0.022],
            [-0.125, 0.000, 0.000, -0.044, 0.000, 0.000, 2.405, 0.000, 0.000],
            [0.000, 0.250, 0.000, 0.000, 0.022, -0.066, 0.000, 2.405, 0.000],
            [0.000, 0.000, -0.125, 0.000, -0.066, 0.022, 0.000, 0.000, 2.405]
        ]
    ) * Au.BOHR**3
    electric_field = jnp.array(
        [
            [0, -0.06404562, -0.06404562],
            [0, -0.02453007, -0.10356116],
            [0, -0.10356116, -0.02453007]
        ]
    ).flatten() * Au.BOHR**2
    mu_tinker = jax.scipy.linalg.solve(t_tinker, electric_field)

    assert jnp.allclose(  # noqa: S101
        output_tinker['tmu'].flatten(),
        t_tinker @ mu_tinker,
        atol=1e-6
    )


def test_energy():
    """Test the energy output of the polarisation model."""
    predicted_energy = output_tinker['polarisation'].sum().item()
    expected_energy = -0.004109111148864031
    assert jnp.allclose(predicted_energy, expected_energy, atol=1e-6)  # noqa: S101


##############
# Water test #
##############

model_water = FENNIX(
    cutoff=5.0,
    rng_key=PRNGKey(0),
    modules={
        'energy': {
            'module_name': 'NEURAL_NET',
            'neurons': [32, 1],
            'input_key': 'coordinates',
        },
        'charges': {
            'module_name': 'CHEMICAL_CONSTANT',
            'value': {
                'O': -0.5,
                'H': 0.25
            },
        },
        'polarisability': {
        'module_name': 'CHEMICAL_CONSTANT',
            'value': {
                'O': 1.0,
                'H': 1.0
            },
        },
        'electric_field': {
            'module_name': 'ELECTRIC_FIELD',
        },
        'polarisation': {
            'module_name': 'POLARISATION'
        },
    }
)


a = -1 / jnp.sqrt(2)**3 + 3 / jnp.sqrt(2)**5
b = -3 / jnp.sqrt(2)**5
c = -1 / jnp.sqrt(2)**3

ao = 1 / 1.0
ah = 1 / 1.0


t_water = jnp.array(
    [
        [ao, 0, 0, 2, 0, 0, -1, 0, 0],
        [0, ao, 0, 0, -1, 0, 0, 2, 0],
        [0, 0, ao, 0, 0, -1, 0, 0, -1],

        [2, 0, 0, ah, 0, 0, a, b, 0],
        [0, -1, 0, 0, ah, 0, b, a, 0],
        [0, 0, -1, 0, 0, ah, 0, 0, c],

        [-1, 0, 0, a, b, 0, ah, 0, 0],
        [0, 2, 0, b, a, 0, 0, ah, 0],
        [0, 0, -1, 0, 0, c, 0, 0, ah]
    ]
) * Au.BOHR**3

coordinates_water = jnp.array(
    [
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]
    ]
).reshape(-1, 3)

output_water = model_water(
        species=species,
        coordinates=coordinates_water,
        natoms=natoms,
        batch_index=batch_index
    )


def test_tmu_water():
    """Test the T@mu output of the polarisation model."""
    electric_field = jnp.array(
        [
            [-0.12585367, -0.12585367, 0.],
            [-0.19055352, -0.06115383, 0.],
            [-0.06115383, -0.19055352, 0.]]
    ).flatten() * Au.BOHR**2
    mu_water = jax.scipy.linalg.solve(t_water, electric_field)

    assert jnp.allclose(  # noqa: S101
        output_water['tmu'].flatten(),
        t_water @ mu_water,
        atol=1e-6
    )
