from .spherical_harmonics import CG_SO3, generate_spherical_harmonics
from .atomic_units import AtomicUnits
from typing import Dict, Any

def deep_update(mapping: Dict[Any, Any], *updating_mappings: Dict[Any, Any]) -> Dict[Any, Any]:
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping