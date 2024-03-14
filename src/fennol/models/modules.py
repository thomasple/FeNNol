from typing import Sequence, Tuple, Dict, Optional, Any
import flax.linen as nn
from inspect import isclass, ismodule
from pkgutil import iter_modules

### python modules where FENNIX Modules are defined ###
from . import misc,physics, embeddings


MODULES: Dict[str, nn.Module] = {}


def available_fennix_modules():
    return list(MODULES.keys())


def register_fennix_module(module: nn.Module, FID: Optional[str] = None):
    if FID is not None:
        name = FID.upper()
    else:
        assert hasattr(module, "FID") and isinstance(
            module.FID, str
        ), "Error: registering a FENNIX module requires setting the FID field as a str."
        name = module.FID.upper()
    if name in MODULES:
        if MODULES[name] == module:
            return
        raise ValueError(
            f"A different module identified as '{name}' is already registered !"
        )
    MODULES[name] = module


def register_fennix_modules(module, recurs=0, max_recurs=2):
    if ismodule(module) and hasattr(module,"__path__"):
        for _, name, _ in iter_modules(module.__path__):
            sub_module = __import__(f"{module.__name__}.{name}", fromlist=[""])
            register_fennix_modules(sub_module, recurs=recurs + 1, max_recurs=max_recurs)
    for m in module.__dict__.values():
        if isclass(m) and issubclass(m, nn.Module) and m != nn.Module:
            if hasattr(m, "FID") and isinstance(m.FID, str):
                register_fennix_module(m)


### REGISTER DEFAULT MODULES #####################
for mods in [misc, physics, embeddings]:
    register_fennix_modules(mods)

##################################################


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
