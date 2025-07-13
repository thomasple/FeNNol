"""
FeNNol Molecular Dynamics Module

This module provides molecular dynamics simulation capabilities using machine learning potentials.
The main entry point is through the `dynamic.py` module which implements the fennol_md command.

Modules:
    dynamic: Main MD simulation engine and parameter documentation
    integrate: Integration schemes and system initialization  
    initial: System setup and model loading
    thermostats: Temperature control algorithms
    barostats: Pressure control algorithms
    colvars: Collective variable calculations
    spectra: Infrared spectrum computation
    utils: Utility functions for restarts and coordinate wrapping

For complete parameter documentation, see `dynamic.py` or use:
    help(fennol.md.dynamic)

Example usage:
    from fennol.md.dynamic import config_and_run_dynamic
    config_and_run_dynamic(Path("input.fnl"))
"""