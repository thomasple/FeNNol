from typing import Sequence, Tuple, Dict, Optional, Any
import flax.linen as nn
from inspect import isclass, ismodule
from pkgutil import iter_modules
import importlib
import importlib.util
import os
import glob

### python modules where FENNIX Modules are defined ###
from . import misc,physics, embeddings,preprocessing


MODULES: Dict[str, nn.Module] = {}
PREPROCESSING: Dict = {}


def available_fennix_modules():
    return list(MODULES.keys())

def available_fennix_preprocessing():
    return list(PREPROCESSING.keys())


def register_fennix_module(module: nn.Module, FID: Optional[str] = None):
    if FID is not None:
        names = [FID.upper()]
    else:
        if not hasattr(module, "FID"):
            print(f"Warning: module {module.__name__} does not have a FID field and no explicit FID was provided. Module was NOT registered.")
        if not (isinstance(
            module.FID, str
        ) or isinstance(module.FID,tuple)):
            print(f"Warning: module {module.__name__} has an invalid FID field. Module was NOT registered.")
        if isinstance(module.FID, str):
            names = [module.FID.upper()]
        else:
            names = [fid.upper() for fid in module.FID]
    for name in names:
        if name in MODULES and MODULES[name] != module:
            raise ValueError(
                f"A different module identified as '{name}' is already registered !"
            )
        MODULES[name] = module

def register_fennix_preprocessing(module, FPID: Optional[str] = None):
    if FPID is not None:
        names = [FPID.upper()]
    else:
        if not hasattr(module, "FPID"):
            print(f"Warning: module {module.__name__} does not have a FPID field and no explicit FPID was provided. Module was NOT registered.")
        if not (isinstance(
            module.FPID, str
        ) or isinstance(module.FPID,tuple)):
            print(f"Warning: module {module.__name__} has an invalid FPID field. Module was NOT registered.")
        if isinstance(module.FPID, str):
            names = [module.FPID.upper()]
        else:
            names = [fid.upper() for fid in module.FPID]
    for name in names:
        if name in PREPROCESSING and PREPROCESSING[name] != module:
            raise ValueError(
                f"A different module identified as '{name}' is already registered !"
            )
        PREPROCESSING[name] = module


def register_fennix_modules(module, recurs=0, max_recurs=2):
    if ismodule(module) and hasattr(module,"__path__"):
        for _, name, _ in iter_modules(module.__path__):
            sub_module = __import__(f"{module.__name__}.{name}", fromlist=[""])
            register_fennix_modules(sub_module, recurs=recurs + 1, max_recurs=max_recurs)
    for m in module.__dict__.values():
        if isclass(m) and issubclass(m, nn.Module) and m != nn.Module:
            if hasattr(m, "FID"):
                register_fennix_module(m)
        elif isclass(m) and hasattr(m, "FPID"):
            register_fennix_preprocessing(m)


### REGISTER DEFAULT MODULES #####################
for mods in [misc, physics, embeddings,preprocessing]:
    register_fennix_modules(mods)
module_path = os.environ.get("FENNOL_MODULES_PATH","").split(":")
for path in module_path:
    if os.path.exists(path):
        for file in glob.glob(f"{path}/*.py"):
            spec = importlib.util.spec_from_file_location("custom_modules", file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            register_fennix_modules(module)

##################################################

def get_modules_documentation():
    doc = {}
    for name, module in MODULES.items():
        doc[name] = module.__doc__
    return doc


class FENNIXModules(nn.Module):
    r"""Sequential module that applies a sequence of FENNIX modules.

    Attributes:
        layers (Sequence[Tuple[nn.Module, Dict]]): Sequence of tuples (module, parameters) to apply.

    """

    layers: Sequence[Tuple[nn.Module, Dict]]

    def __post_init__(self):
        if not isinstance(self.layers, Sequence):
            raise ValueError(
                f"'layers' must be a sequence, got '{type(self.layers).__name__}'."
            )
        super().__post_init__()

    @nn.compact
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.layers:
            raise ValueError(f"Empty Sequential module {self.name}.")

        outputs = inputs
        for layer, prms in self.layers:
            outputs = layer(**prms)(outputs)
        return outputs
