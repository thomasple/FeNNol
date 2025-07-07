import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from ..utils import read_tinker_interval


def build_colvar_distance(colvar_name, colvar_def, global_colvars={}):
    atoms = colvar_def["atoms"]
    assert len(atoms) == 2, f"Colvar {colvar_name} must have exactly 2 atoms defined"
    atom1 = int(atoms[0]) - 1
    atom2 = int(atoms[1]) - 1
    assert atom1 != atom2, f"atom1 and atom2 must be different for colvar {colvar_name}"
    assert (
        atom1 >= 0 and atom2 >= 0
    ), f"atom1 and atom2 must be > 0 for colvar {colvar_name}"

    def colvar_distance(coordinates):
        return jnp.linalg.norm(coordinates[atom1] - coordinates[atom2])

    return colvar_distance


def build_colvar_angle(colvar_name, colvar_def, global_colvars={}):
    atoms = colvar_def["atoms"]
    assert len(atoms) == 3, f"Colvar {colvar_name} must have exactly 3 atoms defined"
    atom1 = int(atoms[0]) - 1
    atom2 = int(atoms[1]) - 1
    atom3 = int(atoms[2]) - 1
    assert (
        atom1 != atom2 and atom2 != atom3 and atom1 != atom3
    ), f"atom1, atom2 and atom3 must be different for colvar {colvar_name}"
    assert (
        atom1 >= 0 and atom2 >= 0 and atom3 >= 0
    ), f"atom1, atom2 and atom3 must be > 0 for colvar {colvar_name}"
    use_radians = colvar_def.get("use_radians", False)
    fact = 1.0 if use_radians else 180.0 / np.pi

    def colvar_angle(coordinates):
        v1 = coordinates[atom1] - coordinates[atom2]
        v2 = coordinates[atom3] - coordinates[atom2]
        v1 = v1 / jnp.linalg.norm(v1)
        v2 = v2 / jnp.linalg.norm(v2)
        return jnp.arccos(jnp.dot(v1, v2)) * fact

    return colvar_angle


def build_colvar_dihedral(colvar_name, colvar_def, global_colvars={}):
    atoms = colvar_def["atoms"]
    assert len(atoms) == 4, f"Colvar {colvar_name} must have exactly 4 atoms defined"
    atom1 = int(atoms[0]) - 1
    atom2 = int(atoms[1]) - 1
    atom3 = int(atoms[2]) - 1
    atom4 = int(atoms[3]) - 1
    # Check that atoms are different and >= 0
    assert (
        atom1 != atom2 and atom2 != atom3 and atom3 != atom4 and atom1 != atom4
    ), f"atom1, atom2, atom3 and atom4 must be different for colvar {colvar_name}"
    assert (
        atom1 >= 0 and atom2 >= 0 and atom3 >= 0 and atom4 >= 0
    ), f"atom1, atom2, atom3 and atom4 must be > 0 for colvar {colvar_name}"
    use_radians = colvar_def.get("use_radians", False)
    fact = 1.0 if use_radians else 180.0 / np.pi

    def colvar_dihedral(coordinates):
        v12 = coordinates[atom2] - coordinates[atom1]
        v23 = coordinates[atom3] - coordinates[atom2]
        v34 = coordinates[atom4] - coordinates[atom3]
        n1 = jnp.cross(v12, v23)
        n2 = jnp.cross(v23, v34)
        n1 = n1 / jnp.linalg.norm(n1)
        n2 = n2 / jnp.linalg.norm(n2)
        cos_phi = (n1*n2).sum()
        sin_phi = (n1*v34).sum()*jnp.linalg.norm(v23)
        phi = jnp.arctan2(sin_phi, cos_phi)
        return phi * fact

    return colvar_dihedral

def build_colvar_maxdcom(colvar_name, colvar_def, global_colvars={}):
    atoms = read_tinker_interval(colvar_def["atoms"])
    
    def colvar_maxdcom(coordinates):
        sel_coord = coordinates[atoms]
        com = jnp.mean(sel_coord, axis=0,keepdims=True)
        return jnp.max(jnp.linalg.norm(sel_coord - com, axis=1))

    return colvar_maxdcom


__RAW_COLVAR = {
    "distance": build_colvar_distance,
    "angle": build_colvar_angle,
    "dihedral": build_colvar_dihedral,
    "maxdcom": build_colvar_maxdcom,
}


def build_colvar_function(colvar_name, colvar_def, global_colvars={}):
    func_def = colvar_def["lambda"]
    assert isinstance(
        func_def, str
    ), f"'lambda' field of colvar '{colvar_name}' must be a string (try using quotation marks)"
    arguments = [s.strip() for s in func_def.split(":")[0].split(",")]
    _cvs = []
    for cv_name in arguments:
        if cv_name in colvar_def:
            cv_def = colvar_def[cv_name]
        elif cv_name in global_colvars:
            cv_def = global_colvars[cv_name]
        else:
            raise ValueError(
                f"Colvar '{colvar_name}' references colvar '{cv_name}' which is not defined in the colvars section"
            )
        cv_type = str(cv_def.get("type", "distance")).lower()
        assert (
            cv_type in __RAW_COLVAR
        ), f"Unknown colvar type '{cv_type}' for colvar '{colvar_name}/{cv_name}'. Available colvars are {list(__RAW_COLVAR.keys())}"
        _cvs.append(__RAW_COLVAR[cv_type](cv_name, cv_def))

    func = eval(
        "lambda " + func_def,
        {
            "__builtins__": None,
            **jax.nn.__dict__,
            **jax.numpy.__dict__,
            **jax.__dict__,
        },
    )

    def colvar_function(coordinates):
        cv_values = []
        for cv in _cvs:
            cv_values.append(cv(coordinates))
        return func(*cv_values)

    return colvar_function


__BUILD_COLVAR = {
    **__RAW_COLVAR,
    "function": build_colvar_function,
}


def setup_colvars(colvars_definitions):
    colvars = {}
    for colvar_name, colvar_def in colvars_definitions.items():
        colvar_type = str(colvar_def.get("type", "distance")).lower()
        assert (
            colvar_type in __BUILD_COLVAR
        ), f"Unknown colvar type '{colvar_type}' for colvar '{colvar_name}'. Available colvars are {list(__BUILD_COLVAR.keys())}"
        colvars[colvar_name] = __BUILD_COLVAR[colvar_type](
            colvar_name, colvar_def, colvars_definitions
        )

    return colvars, list(colvars.keys())
