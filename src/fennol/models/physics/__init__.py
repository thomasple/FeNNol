from .bond import CND4, CNShift, CNStore, SumSwitch
from .dispersion import VdwOQDO
from .electrostatics import ChargeCorrection, Coulomb, QeqD4
from .polarisation import Polarisation
from .repulsion import RepulsionZBL

PHYSICS={
    "REPULSION_ZBL": RepulsionZBL,
    "QEQ_D4": QeqD4,
    "CHARGE_CORRECTION": ChargeCorrection,
    "COULOMB": Coulomb,
    "CN_D4": CND4,
    "CN_SHIFT": CNShift,
    "CN_STORE": CNStore,
    "SUM_SWITCH": SumSwitch,
    "VDW_OQDO": VdwOQDO,
    "POLARISATION": Polarisation,
}