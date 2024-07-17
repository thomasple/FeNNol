import ase
import ase.calculators.calculator
import ase.units
import numpy as np
import jax.numpy as jnp
from . import FENNIX
from .models.preprocessing import convert_to_jax
from typing import Sequence, Union, Optional
from ase.stress import full_3x3_to_voigt_6_stress

class FENNIXCalculator(ase.calculators.calculator.Calculator):
    """FENNIX calculator for ASE.

    Arguments:
    ----------
    model: str or FENNIX
        The path to the model or the model itself.
    use_atom_padding: bool, default=False
        Whether to use atom padding or not. Atom padding is useful to prevent recompiling the model if the number of atoms changes. If the number of atoms is expected to be fixed, it is recommended to set this to False.
    gpu_preprocessing: bool, default=False
        Whether to preprocess the data on the GPU or not. This is useful for large systems, but may not be necessary (or even slower) for small systems.
    atoms: ase.Atoms, default=None
        The atoms object to be used for the calculation. If provided, the calculator will be initialized with the atoms object.
    verbose: bool, default=False
        Whether to print nblist update information or not.
    energy_terms: list of str, default=None
        The energy terms to include in the total energy. If None, this will default to the energy terms defined in the model.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: Union[str, FENNIX],
        gpu_preprocessing: bool = False,
        atoms: Optional[ase.Atoms] = None,
        verbose: bool = False,
        energy_terms: Optional[Sequence[str]] = None,
        float_type: str = "float32",
        **kwargs
    ):
        super().__init__()
        if isinstance(model, FENNIX):
            self.model = model
        else:
            self.model = FENNIX.load(model, **kwargs)
        if energy_terms is not None:
            self.model.set_energy_terms(energy_terms)
        assert float_type in ["float32", "float64"], "float_type must be either 'float32' or 'float64'"
        self.dtype = float_type
        self.gpu_preprocessing = gpu_preprocessing
        self.verbose = verbose
        self._fennol_inputs = None
        if atoms is not None:
            self.preprocess(atoms)

    def calculate(
        self,
        atoms=None,
        properties=["energy"],
        system_changes=ase.calculators.calculator.all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        inputs = self.preprocess(self.atoms, system_changes=system_changes)

        if "stress" in properties:
            e, f, stress, output = self.model._energy_and_forces_and_virial(
                self.model.variables, inputs
            )
            self.results["stress"] = full_3x3_to_voigt_6_stress(np.asarray(stress[0]) * ase.units.Hartree)
            self.results["forces"] = np.asarray(f) * ase.units.Hartree
        elif "forces" in properties:
            e, f, output = self.model._energy_and_forces(self.model.variables, inputs)
            self.results["forces"] = np.asarray(f) * ase.units.Hartree
        else:
            e, output = self.model._total_energy(self.model.variables, inputs)

        self.results["energy"] = float(e[0]) * ase.units.Hartree
        # self.results["fennol_output"] = output

    def preprocess(self, atoms, system_changes=ase.calculators.calculator.all_changes):

        force_cpu_preprocessing = False
        if self._fennol_inputs is None:
            force_cpu_preprocessing = True
            cell = np.asarray(atoms.get_cell(complete=True).array, dtype=self.dtype)
            pbc = np.asarray(atoms.get_pbc(), dtype=bool)
            if np.all(pbc):
                use_pbc = True
            elif np.any(pbc):
                raise NotImplementedError("PBC should be activated in all directions.")
            else:
                use_pbc = False

            species = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
            coordinates = np.asarray(atoms.get_positions(), dtype=self.dtype)
            natoms = np.array([len(species)], dtype=np.int32)
            batch_index = np.array([0] * len(species), dtype=np.int32)

            inputs = {
                "species": species,
                "coordinates": coordinates,
                "natoms": natoms,
                "batch_index": batch_index,
            }
            if use_pbc:
                reciprocal_cell = np.linalg.inv(cell)
                inputs["cells"] = cell.reshape(1, 3, 3)
                inputs["reciprocal_cells"] = reciprocal_cell.reshape(1, 3, 3)
            self._fennol_inputs = convert_to_jax(inputs)
        else:
            if "cell" in system_changes:
                pbc = np.asarray(atoms.get_pbc(), dtype=bool)
                if np.all(pbc):
                    use_pbc = True
                elif np.any(pbc):
                    raise NotImplementedError(
                        "PBC should be activated in all directions."
                    )
                else:
                    use_pbc = False
                if use_pbc:
                    cell = np.asarray(
                        atoms.get_cell(complete=True).array, dtype=self.dtype
                    )
                    reciprocal_cell = np.linalg.inv(cell)
                    self._fennol_inputs["cells"] = jnp.asarray(cell.reshape(1, 3, 3))
                    self._fennol_inputs["reciprocal_cells"] = jnp.asarray(
                        reciprocal_cell.reshape(1, 3, 3)
                    )
                elif "cells" in self._fennol_inputs:
                    del self._fennol_inputs["cells"]
                    del self._fennol_inputs["reciprocal_cells"]
            if "numbers" in system_changes:
                self._fennol_inputs["species"] = jnp.asarray(
                    atoms.get_atomic_numbers(), dtype=jnp.int32
                )
                self._fennol_inputs["natoms"] = jnp.array(
                    [len(self._fennol_inputs["species"])], dtype=np.int32
                )
                self._fennol_inputs["batch_index"] = jnp.array(
                    [0] * len(self._fennol_inputs["species"]), dtype=np.int32
                )
                force_cpu_preprocessing = True
            if "positions" in system_changes:
                self._fennol_inputs["coordinates"] = jnp.asarray(
                    atoms.get_positions(), dtype=self.dtype
                )

        if self.gpu_preprocessing and not force_cpu_preprocessing:
            inputs = self.model.preprocessing.process(
                self.model.preproc_state, self._fennol_inputs
            )
            self.model.preproc_state, state_up, self._fennol_inputs, overflow = (
                self.model.preprocessing.check_reallocate(
                    self.model.preproc_state, inputs
                )
            )
            if self.verbose and overflow:
                print("FENNIX nblist overflow => reallocating nblist")
                print("  size updates:", state_up)
        else:
            self._fennol_inputs = self.model.preprocess(**self._fennol_inputs)

        return self._fennol_inputs
