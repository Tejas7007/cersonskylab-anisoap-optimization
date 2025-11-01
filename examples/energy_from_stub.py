from ase import Atoms
from anisoap_ase.calculator import AniSOAPCalculator
from anisoap_ase.descriptors import anisoap_descriptor
from anisoap_ase.model import LinearModel

atoms = Atoms("H2", positions=[[0,0,0],[0,0,0.74]], pbc=False)

# The stub descriptor returns a 12-length vector → pick 12 weights
w = [0.01] * 12
model = LinearModel(w=w, b=0.0)

atoms.calc = AniSOAPCalculator(descriptor_fn=anisoap_descriptor, model=model)
print("Energy from descriptor+model (eV):", atoms.get_potential_energy())
