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

class Counter:
  def __init__(self,nseg,startsave=1):
    self.i = 0
    self.i_avg = 0
    self.nseg = nseg
    self.startsave = startsave

  @property
  def count(self):
    return self.i
  
  @property
  def count_avg(self):
    return self.i_avg
  
  @property
  def nsample(self):
    return max(self.count_avg-self.startsave+1,1)

  @property
  def is_reset_step(self):
    return self.count == 0
  
  def reset_avg(self):
    self.i_avg=0
  
  def reset_all(self):
    self.i=0
    self.i_avg=0

  def increment(self):
    self.i=self.i+1
    if self.i>=self.nseg:
      self.i=0
      self.i_avg=self.i_avg+1