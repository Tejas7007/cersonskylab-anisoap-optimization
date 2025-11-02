from ase import Atoms
from anisoap_ase.calculator import AniSOAPCalculator

def fake_energy_fn(atoms: Atoms) -> float:
    # deterministic placeholder energy in eV
    return float(len(atoms)) * 0.123

h2 = Atoms("H2", positions=[[0,0,0],[0,0,0.74]], pbc=False)
h2.calc = AniSOAPCalculator(energy_fn=fake_energy_fn)

print("Potential energy (eV):", h2.get_potential_energy())
