from ase import Atoms
from anisoap_ase import AniSOAPCalculator

def test_energy_constant():
    def efn(a: Atoms) -> float: return 0.5
    a = Atoms("He", positions=[[0,0,0]])
    a.calc = AniSOAPCalculator(energy_fn=efn)
    assert abs(a.get_potential_energy() - 0.5) < 1e-12
