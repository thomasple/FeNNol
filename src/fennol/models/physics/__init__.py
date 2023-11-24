from .repulsion import RepulsionZBL
from .electrostatics import QeqD4,ChargeCorrection,Coulomb
from .bond import CND4,CNShift,SumSwitch,CNStore
from .dispersion import VdwOQDO

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
}