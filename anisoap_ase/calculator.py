from __future__ import annotations
from typing import Callable, Optional, Any, Dict, Sequence
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.atoms import Atoms

class AniSOAPCalculator(Calculator):
    """
    Minimal ASE calculator wrapper around AniSOAP-style models.
    Energy-only for now (no forces/stress).

    Provide either:
      - energy_fn(atoms) -> float
      - or (descriptor_fn, model) with energy = model(descriptor_fn(atoms))

    Energies must be in eV.
    """
    implemented_properties = ["energy"]

    def __init__(
        self,
        energy_fn: Optional[Callable[[Atoms], float]] = None,
        descriptor_fn: Optional[Callable[[Atoms], Any]] = None,
        model: Optional[Callable[[Any], float]] = None,
        cache_results: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.energy_fn = energy_fn
        self.descriptor_fn = descriptor_fn
        self.model = model
        self.cache_results = cache_results
        self._last_state: Dict[str, Any] = {}

        if (energy_fn is None) and not (descriptor_fn and model):
            raise ValueError(
                "Provide energy_fn OR (descriptor_fn AND model) to compute energy."
            )

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[Sequence[str]] = None,
        system_changes: Sequence[str] = all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        if properties is None:
            properties = self.implemented_properties

        # hash a minimal state for caching
        state = dict(
            numbers=tuple(self.atoms.get_atomic_numbers()),
            positions=np.asarray(self.atoms.get_positions()).round(12).tobytes(),
            cell=np.asarray(self.atoms.cell.array).round(12).tobytes(),
            pbc=tuple(bool(x) for x in self.atoms.pbc),
        )

        if self.cache_results and self.results and state == self._last_state:
            return

        # compute energy (eV)
        if self.energy_fn is not None:
            energy_ev = float(self.energy_fn(self.atoms))
        else:
            desc = self.descriptor_fn(self.atoms)
            energy_ev = float(self.model(desc))

        # ASE contract
        self.results = {}
        if "energy" in properties:
            self.results["energy"] = energy_ev

        if self.cache_results:
            self._last_state = state

    def get_potential_energy(self, atoms: Optional[Atoms] = None, force_consistent: bool = False) -> float:
        return super().get_potential_energy(atoms=atoms, force_consistent=force_consistent)
