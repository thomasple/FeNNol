import jax
import jax.numpy as jnp
from functools import partial

def colvar_distance(coordinates, atom1, atom2):
    return jnp.linalg.norm(coordinates[atom1] - coordinates[atom2])

def colvar_angle(coordinates, atom1, atom2, atom3):
    v1 = coordinates[atom1] - coordinates[atom2]
    v2 = coordinates[atom3] - coordinates[atom2]
    v1 = v1 / jnp.linalg.norm(v1)
    v2 = v2 / jnp.linalg.norm(v2)
    return jnp.arccos(jnp.dot(v1, v2))

def colvar_dihedral(coordinates, atom1, atom2, atom3, atom4):
    v1 = coordinates[atom1] - coordinates[atom2]
    v2 = coordinates[atom3] - coordinates[atom2]
    v3 = coordinates[atom4] - coordinates[atom3]
    n1 = jnp.cross(v1, v2)
    n2 = jnp.cross(v2, v3)
    n1 = n1 / jnp.linalg.norm(n1)
    n2 = n2 / jnp.linalg.norm(n2)
    return jnp.arccos(jnp.dot(n1, n2))


def setup_colvars(colvars_definitions):
    colvars = {}
    for colvar_name, colvar_def in colvars_definitions.items():
        colvar_type = colvar_def.get("type", "distance")
        if colvar_type == "distance":
            atom1 = colvar_def["atom1"]
            atom2 = colvar_def["atom2"]
            colvars[colvar_name] = partial(colvar_distance, atom1=atom1, atom2=atom2)
        elif colvar_type == "angle":
            atom1 = colvar_def["atom1"]
            atom2 = colvar_def["atom2"]
            atom3 = colvar_def["atom3"]
            colvars[colvar_name] = partial(colvar_angle, atom1=atom1, atom2=atom2, atom3=atom3)
        elif colvar_type == "dihedral":
            atom1 = colvar_def["atom1"]
            atom2 = colvar_def["atom2"]
            atom3 = colvar_def["atom3"]
            atom4 = colvar_def["atom4"]
            colvars[colvar_name] = partial(colvar_dihedral, atom1=atom1, atom2=atom2, atom3=atom3, atom4=atom4)
        else:
            raise ValueError(f"Unknown colvar type {colvar_type}")
    
    return colvars