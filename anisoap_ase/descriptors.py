from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import numpy as np
from ase.atoms import Atoms

class DescriptorError(RuntimeError):
    pass

def atoms_to_numpy(atoms: Atoms) -> Dict[str, np.ndarray]:
    """
    Convert ASE Atoms into numpy payload: positions (Å), atomic numbers, cell, pbc.
    """
    R = atoms.get_positions()          # (N, 3) in Å
    Z = atoms.get_atomic_numbers()     # (N,)
    cell = atoms.cell.array.copy()     # (3, 3) in Å
    pbc = np.asarray(atoms.pbc, dtype=bool)  # (3,)
    return {"R": np.asarray(R, float), "Z": np.asarray(Z, int),
            "cell": np.asarray(cell, float), "pbc": pbc}

def anisoap_descriptor(
    atoms: Atoms,
    *,
    species: Optional[Sequence[int]] = None,
    cutoff: float = 5.0,
    l_max: int = 4,
    n_max: int = 8,
    sigma: float = 0.5,
    implementation: str = "auto",
    **kwargs: Any,
) -> Any:
    """
    Compute (or stub) AniSOAP descriptors for `atoms`.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic configuration.
    species : sequence[int] | None
        Allowed atomic numbers. If None, inferred from atoms.
    cutoff : float
        Radial cutoff (Å) for local environments.
    l_max, n_max, sigma : int/float
        Typical SOAP/AniSOAP hyperparameters.
    implementation : {"auto","python","rust","torch"}
        Choose backend if your AniSOAP package exposes multiple.
    **kwargs : Any
        Forwarded to the underlying AniSOAP call.

    Returns
    -------
    desc : Any
        Descriptor object/array accepted by your energy model.
    """
    payload = atoms_to_numpy(atoms)

    # infer species if not provided
    if species is None:
        species = sorted(set(int(z) for z in payload["Z"]))

    # --- Replace this try/except with your real AniSOAP call when ready ---
    try:
        # from anisoap_core import compute_descriptors
        # desc = compute_descriptors(
        #     R=payload["R"], Z=payload["Z"],
        #     cell=payload["cell"], pbc=payload["pbc"],
        #     species=species, cutoff=cutoff, l_max=l_max, n_max=n_max,
        #     sigma=sigma, implementation=implementation, **kwargs
        # )
        # return desc
        raise ImportError  # placeholder until real backend import is wired
    except ImportError:
        # --- Stub: simple deterministic descriptor (not scientific) ---
        R = payload["R"]
        Z = payload["Z"].astype(float)
        r = np.linalg.norm(R, axis=1, keepdims=True)
        per_atom = np.hstack([Z[:, None], R, r, np.ones_like(r)])  # shape (N,6)
        summed = per_atom.sum(axis=0)   # (6,)
        meaned = per_atom.mean(axis=0)  # (6,)
        desc = np.concatenate([summed, meaned], axis=0)  # (12,)
        return desc
