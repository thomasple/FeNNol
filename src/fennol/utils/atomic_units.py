from math import pi
import math
from typing import Dict, Union, Optional
from .unit_parser import parse_unit_string


class _AtomicUnits:
    """Class to hold atomic unit constants and conversions.
    Definition:
        - e = 1. (elementary charge)
        - hbar = 1. (reduced Planck's constant)
        - me = 1. (electron mass in atomic units)
        - a0 = 1. (bohr radius)
    """

    LIGHTSPEED = 137.035999177  # Speed of light in atomic units
    FSC = 1.0 / LIGHTSPEED  # Fine structure constant
    HBAR = 1.0  # Reduced Planck's constant
    PLANCK = 2.0 * pi * HBAR  # Planck's constant (not used directly)
    ME = 1.0  # Electron mass in atomic units (AU)
    MPROT = 1836.1526734252586  # proton/electron mass ratio
    NA = 6.02214129e23  # Avogadro's number
    AVOGADRO = NA  # Alias for Avogadro's number
    MOL = 1.0 / NA  # number of particules to Mole
    EPS0 = 1.0 / (4.0 * pi)  # Vacuum permittivity in atomic units
    K_E = 1.0  # Coulomb's constant in atomic units (k_e = 1/(4πε₀) = 1 in atomic units)
    COULOMB = K_E  # Alias for backward compatibility
    A0 = 1.0  # Bohr radius in atomic units (AU)
    K_B = 3.166811563e-6  # Boltzmann constant in atomic units Ha/K (AU)

    CONSTANTS = {
        "LIGHTSPEED": (LIGHTSPEED, {"L": 1, "T": -1}),
        "FSC": (FSC, {}),
        "HBAR": (HBAR, {"E": 1, "T": 1}),
        "PLANCK": (PLANCK, {"E": 1, "T": 1}),
        "ME": (ME, {"M": 1}),
        "MPROT": (MPROT, {"M": 1}),
        "NA": (NA, {}),
        "AVOGADRO": (NA, {}),
        "EPS0": (EPS0, {"E": -1, "L": -1}),
        "K_E": (K_E, {"E": 1, "L": 1}),
        "COULOMB": (COULOMB, {"E": 1, "L": 1}),
        "A0": (A0, {"L": 1}),
        "K_B": (K_B, {"E": 1}),
    }

    ### LENGTH UNITS
    ANGSTROM = 0.52917721  # Bohr to Angstrom
    ANG = ANGSTROM  # Alias for ANGSTROM
    NM = ANGSTROM * 1e-1  # 1 nm = 10 Angstroms
    CM = ANGSTROM * 1e-8
    LENGTH_UNITS = {
        "A0": 1.0,
        "BOHR": 1.0,
        "ANGSTROM": ANGSTROM,
        "ANG": ANGSTROM,
        "A": ANGSTROM,
        "NM": NM,
        "CM": CM,
    }

    ### TIME UNITS
    FS = 2.4188843e-2  # AU time to femtoseconds
    PS = FS / 1000  # AU time to picoseconds
    NS = PS / 1000  # AU time to nanoseconds
    TIME_UNITS = {
        "AU_T": 1.0,
        "FS": FS,
        "FEMTOSECONDS": FS,
        "PS": PS,
        "PICOSECONDS": PS,
        "NS": NS,
        "NANOSECONDS": NS,
    }

    ### ENERGY UNITS
    EV = 27.211386024367243  # Hartree to eV
    KCALPERMOL = 627.5096080305927  # Hartree to kcal/mol
    KJPERMOL = 2625.5002  # Hartree to kJ/mol
    RY = 2.0  # Hartree to Rydberg
    KELVIN = 1.0 / K_B  # Hartree to Kelvin (K/Ha)
    KCAL = KCALPERMOL * MOL
    KJ = KJPERMOL * MOL
    ENERGY_UNITS = {
        "HARTREE": 1.0,
        "HA": 1.0,
        "EV": EV,
        "MEV": 1.0e3 * EV,  # milli electronvolt
        "KCALPERMOL": KCALPERMOL,
        "KJPERMOL": KJPERMOL,
        "RY": RY,
        "KELVIN": KELVIN,
        "KCAL": KCAL,
        "KJ": KJ,
    }

    # MASS UNITS
    DA = 1.0 / MPROT  # amu to Dalton (~ g/mol)
    GRAM = DA * MOL  # amu to gram
    MASS_UNITS = {
        "AU_M": 1.0,
        "ME": 1.0,
        "DA": DA,
        "GRAM": GRAM,
        "KG": GRAM * 1e-3,
        "KILOGRAM": GRAM * 1e-3,
    }

    ### OTHER UNITS
    DEBYE = 2.541746  # e.Bohr to Debye

    # FREQUENCY UNITS
    THZ = 1000.0 / FS  # AU frequency to THz
    CM1 = 219471.52  # AU angular frequency to cm-1 (spectroscopist's wavenumber)

    # PRESSURE UNITS
    KBAR = 294210.2648438959  # Hartree/bohr**3 to kbar
    ATM = KBAR * 1000.0 / 1.01325  # Hartree/bohr**3 to atm
    GPA = 0.1 * KBAR  # Hartree/bohr**3 to GPa

    # FORCE UNITS
    NNEWTON = 82.387  # Ha/bohr to nNewton

    OTHER_UNITS = {
        "MOL": (MOL, {}),
        "DEBYE": (DEBYE, {"L": 1}),
        "THZ": (THZ, {"T": -1}),
        "CM-1": (CM1, {"T": -1}),
        "CM1": (CM1, {"T": -1}),
        "KBAR": (KBAR, {"L": -3, "E": 1}),
        "ATM": (ATM, {"L": -3, "E": 1}),
        "GPA": (GPA, {"L": -3, "E": 1}),
        "NNEWTON": (NNEWTON, {"L": -1, "E": 1}),
    }

    def __setattr__(self, name, value):
        raise AttributeError(
            f"_AtomicUnits is immutable. Cannot set attribute '{name}'."
        )


class UnitSystem:
    """Class to handle unit systems and conversions.
    A unit system is defined by its base units which are a set of THREE units among:
        - Length (L)
        - Time (T)
        - Energy (E)
        - Mass (M)
    that are defined at initialization.
    In all unit systems, we use the elementary charge and Kelvin as base units.

    Other units are derived from these base units.

    Unit multipliers are made accessible as attributes of the class.
    They correspond to the conversion factors from the unit system to the unit.
    Example:
        us = UnitSystem(L='BOHR', T='AU_T', E='HARTREE')
        print(us.BOHR) # 1.0
        print(us.ANGSTROM) # 0.52917721
        print(au.FS) # 2.4188843e-2

        us = UnitSystem(L='ANGSTROM', E='EV',M='DA')
        print(us.BOHR)  # 1.889725989
        print(us.fs) # 10.217477

    WARNING: This is the opposite convention as ASE units.

    shorthand notations:
        - au: Atomic Units
        - us: Unit/User System

    """

    _initialized = False

    def __init__(
        self,
        L: Optional[str] = None,
        T: Optional[str] = None,
        E: Optional[str] = None,
        M: Optional[str] = None,
    ):

        self.unit_system = {}
        self.us_to_au = {}
        if L is not None:
            L = L.strip().upper()
            self.unit_system["L"] = L
            self.us_to_au["L"] = 1.0 / _AtomicUnits.LENGTH_UNITS[L]
        if T is not None:
            T = T.strip().upper()
            self.unit_system["T"] = T
            self.us_to_au["T"] = 1.0 / _AtomicUnits.TIME_UNITS[T]
        if E is not None:
            E = E.strip().upper()
            self.unit_system["E"] = E
            self.us_to_au["E"] = 1.0 / _AtomicUnits.ENERGY_UNITS[E]
        if M is not None:
            M = M.strip().upper()
            self.unit_system["M"] = M
            self.us_to_au["M"] = 1.0 / _AtomicUnits.MASS_UNITS[M]
        assert (
            len(self.unit_system) == 3
        ), f"Unit system must have exactly 3 base units to be consistent. Got: {self.unit_system}"

        # Derive the last base unit from the other ones.
        if L is None:
            self.us_to_au["L"] = self.base_us2au_converter(E=0.5, T=1, M=-0.5)
        elif T is None:
            self.us_to_au["T"] = self.base_us2au_converter(E=-0.5, L=1, M=0.5)
        elif E is None:
            self.us_to_au["E"] = self.base_us2au_converter(L=2, T=-2, M=1)
        elif M is None:
            self.us_to_au["M"] = self.base_us2au_converter(E=1, T=2, L=-2)

        self.us_to_unit = {}
        ## LENGTH UNITS
        for unit, au_to_unit in _AtomicUnits.LENGTH_UNITS.items():
            self.us_to_unit[unit.upper()] = self.us_to_au["L"] * au_to_unit

        ## TIME UNITS
        for unit, au_to_unit in _AtomicUnits.TIME_UNITS.items():
            self.us_to_unit[unit.upper()] = self.us_to_au["T"] * au_to_unit

        ## ENERGY UNITS
        for unit, au_to_unit in _AtomicUnits.ENERGY_UNITS.items():
            self.us_to_unit[unit.upper()] = self.us_to_au["E"] * au_to_unit

        ## MASS UNITS
        for unit, au_to_unit in _AtomicUnits.MASS_UNITS.items():
            self.us_to_unit[unit.upper()] = self.us_to_au["M"] * au_to_unit

        ### OTHER UNITS
        for unit, (au_to_unit, powers) in _AtomicUnits.OTHER_UNITS.items():
            self.us_to_unit[unit.upper()] = au_to_unit * self.base_us2au_converter(
                **powers
            )

        ##CONSTANTS
        self.CONSTANTS = {}
        for const_name, (value_au, powers) in _AtomicUnits.CONSTANTS.items():
            self.CONSTANTS[const_name.upper()] = value_au / self.base_us2au_converter(
                **powers
            )

        # ## SET ATTRIBUTES FOR EASY ACCESS
        for const_name, value in self.CONSTANTS.items():
            self.__setattr__(const_name, value)
        for unit, value in self.us_to_unit.items():
            self.__setattr__(unit, value)

        self.UNITS_LIST = list(self.us_to_unit.keys())

        # Add some common constants for get_multiplier
        self.us_to_unit["1"] = 1.0
        self.us_to_unit["2PI"] = 2.0 * pi

        self._initialized = True

    def __setattr__(self, name, value):
        if not self._initialized:
            super().__setattr__(name, value)
        else:
            raise AttributeError(
                f"UnitSystem is immutable after initialization. Cannot set attribute '{name}'."
            )

    def list_units(self):
        """List all units in the unit system."""
        return self.UNITS_LIST

    def list_constants(self):
        """List all constants in the unit system."""
        return list(self.CONSTANTS.keys())

    def base_us2au_converter(self, **powers: Dict[str, Union[int, float]]):
        """
        Get the conversion multiplier from the unit system to atomic units based on powers of base units.
        """
        return math.prod(
            [self.us_to_au[unit.upper()] ** power for unit, power in powers.items()]
        )

    def base_au2us_converter(self, **powers: Dict[str, Union[int, float]]):
        """
        Get the conversion multiplier from atomic units to the unit system based on powers of base units.
        """
        return 1.0 / self.base_us2au_converter(**powers)

    def get_multiplier(self, unit_string: str) -> float:
        """
        Parse a unit string and return the conversion multiplier from the unit system to that unit.

        Example:
          us = UnitSystem('BOHR', 'AU_T', 'HA')
          multiplier = us.get_multiplier('ANGSTROM')
          print(multiplier)  # Should print 0.5291.. (the bohr radius in Angstroms)
          multiplier = us.get_multiplier('EV')
          print(multiplier)  # Should print 27.211386024367243

        Supports syntax like:
        - Simple units: "EV", "ANGSTROM"
        - Powers: "ANGSTROM^{2}", "FS^{-1}", "ANGSTROM^2"
        - Products: "EV*ANGSTROM", "KBAR*FS"
        - Quotients: "EV/ANGSTROM", "KBAR/FS"
        - Complex: "EV*ANGSTROM^{2}/FS^{3}"
        - Floating point powers: "ANGSTROM^{2.5}"
        - Parentheses: "(EV*ANGSTROM)^2", "EV/(ANGSTROM*FS)"
        """
        return parse_unit_string(unit_string, self.us_to_unit)


AtomicUnits = au = UnitSystem(L="BOHR", T="AU_T", E="HARTREE")
