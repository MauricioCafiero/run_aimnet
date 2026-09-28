"""AIMNet2 as an ASE calculator front-end for small molecules.

This module wraps the AIMNet2 foundation models from
`aimnetcentral <https://github.com/isayevlab/aimnetcentral>`_ and exposes them
through ASE for the four classic atomistic tasks:

    * single-point energy / forces / partial charges / dipole
    * geometry optimization
    * vibrational analysis (harmonic frequencies)
    * molecular dynamics (NVE / NVT)

plus SMILES/XYZ input helpers, so a molecule can go from a SMILES string to a
relaxed geometry and frequencies in three calls.

It is the AIMNet2 sibling of the ``mace_calc`` module, with the same function
names and signatures. Two things differ, both because of what AIMNet2 is:

* **Charge is a first-class input.** AIMNet2 is charge-aware: it takes the total
  molecular charge and predicts per-atom partial charges. Every task function
  here accepts ``charge=``, and :func:`smiles_to_atoms` reads the formal charge
  straight off the SMILES, so anions/cations Just Work. MACE has no such knob.
* **There is no auto-escalation to a bigger model.** AIMNet2 covers exactly 14
  elements (H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I) and there is no
  metal-capable sibling to fall back to, so an unsupported element raises a
  clear error naming the offending symbols instead of silently switching model.

Multiplicity is only meaningful for the spin-aware ``"nse"`` model; the other
models are closed-shell and ignore ``mult``.

.. warning::
   **Never compare energies across model families.** ``"wb97m"``, ``"b973c"``,
   ``"nse"`` and ``"rxn"`` are trained to different targets and, in the case of
   ``"rxn"``, a learned shifted-electronic scale -- ethanol comes out at
   -4221.6 eV under ``"wb97m"`` but -1.1 eV under ``"rxn"``. Energy differences
   are only meaningful within one ``model=``. Ensemble members of the same
   family ("wb97m-1" ... "wb97m-3") *are* comparable to each other.

Energies are in eV, lengths in Angstrom, forces in eV/Angstrom -- ASE's native
units, which is what AIMNet2ASE already returns.

Runs on the CPU by default (Apple Silicon has no CUDA); ``device="cuda"`` is
honoured where available.

Example
-------
>>> from aimnet_calc import from_smiles, optimize, singlepoint, vibrations
>>> atoms = from_smiles("CCO")            # ethanol, charge read from SMILES
>>> optimize(atoms, fmax=0.01)
>>> res = singlepoint(atoms)
>>> res["energy"], res["charges"][:3]
>>> vibrations(atoms)
>>> # a charged species:
>>> acetate = from_smiles("CC(=O)[O-]")   # atoms.info["charge"] == -1
>>> singlepoint(acetate)["energy"]
"""

from __future__ import annotations

import os

# --- macOS / Apple-Silicon stability guard (must precede the torch import) ---
# torch and (if it is ever imported alongside) tblite each ship a copy of
# libomp; loading both aborts with "OMP: Error #15" unless this is allowed.
# Unlike the MACE stack, AIMNet2 does not use e3nn, so the OMP_NUM_THREADS=1
# segfault workaround is NOT needed here -- threads are left at the system
# default. Pin OMP_NUM_THREADS yourself before import if you want to.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from ase import Atoms, units
from ase.io import Trajectory, read, write
from ase.optimize import BFGS, FIRE, LBFGS
from ase.vibrations import Vibrations
from ase.md import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet

log = logging.getLogger("aimnet_calc")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Friendly alias -> upstream model key (see aimnet/calculators/model_registry.yaml).
# The upstream keys ending in _0 are the first member of a 4-model ensemble;
# members _1.._3 are reachable via the "<alias>-<i>" aliases below.
MODEL_ALIASES: dict[str, str] = {
    # wB97M-D3 -- the reference-quality default, same functional family as
    # MACE-OFF23. Best general-purpose choice.
    "wb97m": "aimnet2-wb97m-d3_0",
    # B97-3c -- cheaper training level, geometries close to B3LYP.
    "b973c": "aimnet2-b973c-d3_0",
    # The 2025 B97-3c retrain; the variant UMADock defaults to.
    "2025": "aimnet2-b973c-2025-d3_0",
    # Spin-aware (non-singlet) model -- the only one that honours `mult`.
    "nse": "aimnet2-nse_0",
    # Reaction-path model (bond breaking/forming).
    "rxn": "aimnet2-rxn_0",
}
# Ensemble members: "wb97m-1" ... "wb97m-3", etc. An ensemble of all four gives
# a spread that is a cheap uncertainty estimate.
for _base, _key in list(MODEL_ALIASES.items()):
    if _key.endswith("_0"):
        for _i in (1, 2, 3):
            MODEL_ALIASES[f"{_base}-{_i}"] = _key[:-2] + f"_{_i}"

DEFAULT_MODEL = "wb97m"

# --- Element coverage --------------------------------------------------------
# AIMNet2 supports exactly H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I.
# NOTE: no metals at all -- not even Na/K/Mg/Ca counterions.
_AIMNET2_ELEMENTS = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53})


def aimnet2_elements() -> frozenset[int]:
    """Atomic numbers supported by AIMNet2 (H,B,C,N,O,F,Si,P,S,Cl,As,Se,Br,I)."""
    return _AIMNET2_ELEMENTS


def check_elements(atoms: Atoms) -> None:
    """Raise ``ValueError`` if ``atoms`` contains an element AIMNet2 cannot do.

    AIMNet2 has no metal-capable sibling model, so there is nothing to escalate
    to -- an unsupported element is a hard stop, and the message names it.
    """
    from ase.data import chemical_symbols

    bad = sorted(set(int(z) for z in atoms.numbers) - _AIMNET2_ELEMENTS)
    if bad:
        symbols = ", ".join(chemical_symbols[z] for z in bad)
        supported = ", ".join(chemical_symbols[z] for z in sorted(_AIMNET2_ELEMENTS))
        raise ValueError(
            f"AIMNet2 does not support: {symbols}. Supported elements are "
            f"{supported} (no metals). Use a MACE-OMOL calculator for these."
        )


def list_models() -> list[str]:
    """Available model aliases, in registry order."""
    return list(MODEL_ALIASES)


def _resolve_alias(model: str) -> str:
    """Map a friendly alias to the upstream model key (pass-through if unknown)."""
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    # Allow raw upstream keys/aliases ("aimnet2", "aimnet2-2025", ...) verbatim.
    return model


def _pick_device(device: str) -> str:
    """Resolve ``"auto"``; MPS is not supported by AIMNet2, so CPU is the
    Apple-Silicon path."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Calculator construction
# ---------------------------------------------------------------------------
def get_calculator(
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
    charge: int = 0,
    mult: int = 1,
    validate_species: bool = True,
    **kwargs,
):
    """Build an ASE-compatible AIMNet2 calculator.

    Parameters
    ----------
    model : str
        Alias from :func:`list_models` (``"wb97m"`` default, ``"b973c"``,
        ``"2025"``, ``"nse"``, ``"rxn"``, or an ensemble member like
        ``"wb97m-2"``). Raw upstream model keys are passed through unchanged.
    device : str
        ``"cpu"`` (default; the only option on Apple Silicon -- AIMNet2 has no
        MPS support), ``"cuda"``, or ``"auto"``.
    charge : int
        Total molecular charge. AIMNet2 is charge-aware, so this genuinely
        changes the energy. Overridden per-structure by ``atoms.info["charge"]``.
    mult : int
        Spin multiplicity (2S+1). Only the ``"nse"`` model uses it.
    validate_species : bool
        Let AIMNet2 check the element set too (belt and braces alongside
        :func:`check_elements`).

    Model weights are downloaded on first use and cached under ``~/.cache/aimnet``.
    """
    from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator

    key = _resolve_alias(model)
    device = _pick_device(device)
    base = AIMNet2Calculator(key, device=device, **kwargs)
    calc = AIMNet2ASE(base, charge=charge, mult=mult, validate_species=validate_species)
    log.info("Loaded AIMNet2 model %s (%s, %s, charge=%d)", model, key, device, charge)
    return calc


def attach(atoms: Atoms, model: str = DEFAULT_MODEL, charge: Optional[int] = None,
           **calc_kw) -> Atoms:
    """Attach an AIMNet2 calculator to ``atoms`` and return ``atoms`` (in-place).

    ``charge`` defaults to ``atoms.info["charge"]`` when present (which is what
    :func:`smiles_to_atoms` sets from the SMILES), else 0.
    """
    check_elements(atoms)
    if charge is None:
        charge = int(atoms.info.get("charge", 0))
    atoms.info.setdefault("charge", charge)
    # Record the spin state too, so it survives a later re-attach. Only the
    # "nse" model reads it; the closed-shell models ignore it.
    if "mult" in calc_kw:
        atoms.info["mult"] = int(calc_kw["mult"])
    elif "mult" in atoms.info:
        calc_kw["mult"] = int(atoms.info["mult"])
    atoms.calc = get_calculator(model=model, charge=charge, **calc_kw)
    return atoms


def _is_aimnet_calc(calc) -> bool:
    return "AIMNet2" in type(calc).__name__


def _ensure_calc(atoms: Atoms, model: str, charge: Optional[int], calc_kw: dict) -> None:
    if atoms.calc is None or not _is_aimnet_calc(atoms.calc):
        attach(atoms, model=model, charge=charge, **calc_kw)


# ---------------------------------------------------------------------------
# Task 1: single point
# ---------------------------------------------------------------------------
def singlepoint(
    atoms: Atoms,
    model: str = DEFAULT_MODEL,
    charge: Optional[int] = None,
    stress: bool = False,
    **calc_kw,
) -> dict:
    """Run a single-point calculation.

    Returns a dict with ``energy`` (eV), ``forces`` (N,3; eV/Angstrom),
    ``charges`` (N; AIMNet2's predicted partial charges, e) and ``dipole``
    (3; e*Angstrom) -- the last two are AIMNet2 extras MACE does not provide.
    ``stress`` (eV/Angstrom^3) is added if requested and the system is periodic.
    """
    _ensure_calc(atoms, model, charge, calc_kw)
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces())
    out = {"energy": energy, "forces": forces}
    # Partial charges and dipole come for free from the same forward pass.
    try:
        out["charges"] = np.asarray(atoms.get_charges())
        out["dipole"] = np.asarray(atoms.get_dipole_moment())
    except Exception as exc:  # pragma: no cover
        log.warning("charges/dipole unavailable: %s", exc)
    if stress:
        if not atoms.pbc.any():
            log.warning("stress requested but system is non-periodic; skipping.")
        else:
            out["stress"] = np.asarray(atoms.get_stress())
    return out


# ---------------------------------------------------------------------------
# Task 2: geometry optimization
# ---------------------------------------------------------------------------
_OPTIMIZERS = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}


def optimize(
    atoms: Atoms,
    model: str = DEFAULT_MODEL,
    charge: Optional[int] = None,
    fmax: float = 0.01,  # eV/Angstrom
    steps: int = 500,
    optimizer: str = "FIRE",
    trajectory: Optional[Union[str, Path]] = None,
    logfile: Optional[Union[str, Path]] = "-",
    **calc_kw,
) -> dict:
    """Relax the geometry to ``fmax`` (eV/Angstrom).

    Returns a dict with the relaxed ``energy``, final ``forces``, and whether
    the optimizer converged. The ``atoms`` object is updated in place.
    """
    _ensure_calc(atoms, model, charge, calc_kw)
    try:
        Opt = _OPTIMIZERS[optimizer]
    except KeyError as exc:
        raise ValueError(
            f"Unknown optimizer {optimizer!r}. Choose from {list(_OPTIMIZERS)}."
        ) from exc
    dyn = Opt(atoms, trajectory=str(trajectory) if trajectory else None, logfile=logfile)
    converged = dyn.run(fmax=fmax, steps=steps)
    return {
        "energy": float(atoms.get_potential_energy()),
        "forces": np.asarray(atoms.get_forces()),
        "converged": bool(converged),
        "fmax": float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
    }


# ---------------------------------------------------------------------------
# Task 3: vibrations
# ---------------------------------------------------------------------------
def vibrations(
    atoms: Atoms,
    model: str = DEFAULT_MODEL,
    charge: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
    nfree: int = 2,
    delta: float = 0.01,  # Angstrom
    name: str = "vib",
    **calc_kw,
) -> dict:
    """Harmonic vibrational analysis by finite differences.

    Returns a dict with ``frequencies_cm1`` (ASE convention: imaginary modes are
    reported as negative), ``energies_meV``, and the vib data name.

    Notes
    -----
    Run on a relaxed geometry (``optimize`` first) or you will get spurious
    imaginary modes. AIMNet2 evaluates in float32 internally -- there is no
    float64 knob as in MACE -- so tighten ``fmax`` in the preceding optimization
    rather than expecting float64-grade gradients here. For a cleaner result,
    :func:`hessian` gives the analytic second derivatives instead.
    """
    _ensure_calc(atoms, model, charge, calc_kw)
    vib = Vibrations(atoms, indices=indices, nfree=nfree, delta=delta, name=name)
    # ASE caches per-displacement results and skips recomputation if present.
    # Stale/partial files (e.g. from a killed run) make get_frequencies crash
    # on None forces, so wipe the cache before a fresh run.
    try:
        vib.clean()
    except Exception:  # pragma: no cover
        pass
    vib.run()
    freqs = np.asarray(vib.get_frequencies())  # cm^-1
    energies = np.asarray(vib.get_energies())  # eV
    try:
        vib.summary()
    except Exception:  # pragma: no cover
        pass
    return {
        "frequencies_cm1": freqs,
        "energies_meV": energies * 1000.0,
        "name": name,
        "nmodes": len(freqs),
    }


def hessian(
    atoms: Atoms,
    model: str = DEFAULT_MODEL,
    charge: Optional[int] = None,
    **calc_kw,
) -> np.ndarray:
    """Cartesian Hessian, (3N, 3N) in eV/Angstrom^2.

    AIMNet2 gets this by double-backward through its own energy graph, so it is
    exact rather than a finite difference -- an advantage over the MACE path.
    Non-periodic systems only, and memory grows sharply past ~100 atoms.
    """
    _ensure_calc(atoms, model, charge, calc_kw)
    return np.asarray(atoms.calc.get_hessian(atoms))


# ---------------------------------------------------------------------------
# Task 4: molecular dynamics
# ---------------------------------------------------------------------------
def run_md(
    atoms: Atoms,
    model: str = DEFAULT_MODEL,
    charge: Optional[int] = None,
    T_K: float = 300.0,
    timestep_fs: float = 0.5,
    steps: int = 1000,
    ensemble: str = "nvt",
    friction: float = 0.01,  # 1/fs, Langevin
    trajectory: Optional[Union[str, Path]] = None,
    traj_interval: int = 10,
    logfile: Optional[Union[str, Path]] = "-",
    loginterval: int = 100,
    seed: Optional[int] = None,
    **calc_kw,
) -> dict:
    """Run a short MD trajectory.

    Parameters
    ----------
    ensemble : str
        ``"nvt"`` (Langevin, default) or ``"nve"`` (velocity Verlet).
    trajectory : path
        If given, an ASE ``Trajectory`` is written every ``traj_interval`` steps.
    seed : int, optional
        Seed for the Maxwell-Boltzmann velocity distribution.
    """
    _ensure_calc(atoms, model, charge, calc_kw)

    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_K, rng=rng, force_temp=True)
    Stationary(atoms)  # remove total momentum
    ZeroRotation(atoms)  # remove angular momentum

    dt = timestep_fs * units.fs

    if ensemble.lower() == "nvt":
        dyn = Langevin(atoms, dt, temperature_K=T_K, friction=friction / units.fs,
                       logfile=logfile, loginterval=loginterval, rng=rng)
    elif ensemble.lower() == "nve":
        dyn = VelocityVerlet(atoms, dt, logfile=logfile, loginterval=loginterval)
    else:
        raise ValueError(f"Unknown ensemble {ensemble!r}; use 'nvt' or 'nve'.")

    if trajectory:
        traj = Trajectory(str(trajectory), "w", atoms)
        dyn.attach(traj.write, interval=traj_interval)

    dyn.run(steps)

    return {
        "ensemble": ensemble,
        "T_K": T_K,
        "steps": steps,
        "timestep_fs": timestep_fs,
        "final_energy": float(atoms.get_potential_energy()),
        "trajectory": str(trajectory) if trajectory else None,
    }


# ---------------------------------------------------------------------------
# Convenience: build inputs
# ---------------------------------------------------------------------------
def molecule(name: str, **kw) -> Atoms:
    """Build a small molecule from the ASE database (e.g. 'H2O', 'CH3OH')."""
    from ase.build import molecule as _molecule

    return _molecule(name, **kw)


def from_xyz(path: Union[str, Path], charge: int = 0) -> Atoms:
    """Read a single geometry from an XYZ file.

    ``charge`` is stored in ``atoms.info`` so the task functions pick it up --
    XYZ carries no charge of its own, so pass it here for an ion.
    """
    atoms = read(str(path))
    atoms.info["charge"] = int(charge)
    return atoms


def smiles_to_atoms(
    smiles: str,
    n_conformers: int = 5,
    seed: int = 42,
    forcefield: str = "mmff",
    minimize_iters: int = 500,
    prune_rms: float = 0.5,
) -> Atoms:
    """Build a 3D molecule from a SMILES string via an RDKit conformer search.

    Pipeline: parse -> add Hs -> embed ``n_conformers`` (ETKDGv3, RMS-pruned) ->
    minimize each with MMFF (falls back to UFF if MMFF is unavailable for the
    element set) -> return the lowest-energy conformer as an ASE ``Atoms``
    object (positions in Angstrom).

    The molecule's **formal charge is read off the SMILES** and stored in
    ``atoms.info["charge"]``, so a charged species is handed to AIMNet2 with the
    right total charge without any extra argument.

    Requires the optional ``rdkit`` dependency (``uv pip install rdkit``).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "smiles_to_atoms requires rdkit. Install with: "
            "uv pip install rdkit (or pip install rdkit)."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    charge = Chem.GetFormalCharge(mol)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    if not cids:
        raise RuntimeError(f"RDKit failed to embed any conformers for {smiles!r}")

    ff_name = forcefield.lower()
    energies: list[tuple[float, int, str]] = []  # (energy, conf_id, method)
    for cid in cids:
        method_used = ff_name
        if ff_name == "mmff":
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is None:
                # MMFF not parameterized for this element set -> use UFF
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                method_used = "uff"
            else:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            method_used = "uff"
        ff.Minimize(maxIts=minimize_iters)
        energies.append((ff.CalcEnergy(), cid, method_used))

    best_e, best_cid, best_method = min(energies)
    log.info("smiles_to_atoms: %d conformers, lowest E=%.3f (%s) confId=%d charge=%d",
             len(energies), best_e, best_method, best_cid, charge)

    conf = mol.GetConformer(best_cid)
    from ase import Atom

    atoms = Atoms(
        [Atom(atom.GetSymbol(), conf.GetAtomPosition(i))
         for i, atom in enumerate(mol.GetAtoms())]
    )
    atoms.info["smiles"] = smiles
    atoms.info["charge"] = charge
    atoms.info["conformer_energy"] = best_e
    atoms.info["forcefield"] = best_method
    atoms.info["n_conformers_sampled"] = len(energies)
    return atoms


def from_smiles(smiles: str, n_conformers: int = 5, seed: int = 42, **kw) -> Atoms:
    """Build a 3D molecule from SMILES (multi-conformer MMFF). See
    :func:`smiles_to_atoms`."""
    return smiles_to_atoms(smiles, n_conformers=n_conformers, seed=seed)


def smiles_to_xyz(
    smiles: str,
    path: Union[str, Path],
    n_conformers: int = 5,
    seed: int = 42,
    comment: Optional[str] = None,
    **kw,
) -> Path:
    """SMILES -> lowest-energy MMFF conformer -> XYZ file ready for AIMNet2.

    Writes a single-geometry XYZ (ASE format) at ``path`` and returns the path.
    Note that XYZ cannot carry the charge: for an ion, pass the same charge to
    :func:`from_xyz` when reading it back. The comment line records it.
    """
    atoms = smiles_to_atoms(smiles, n_conformers=n_conformers, seed=seed)
    c = comment or (
        f"{atoms.get_chemical_formula()} from SMILES; charge={atoms.info['charge']}; "
        f"ff={atoms.info.get('forcefield')}"
    )
    write(str(path), atoms, format="xyz", comment=c)
    log.info("smiles_to_xyz: wrote %s (%s, charge=%d)", path,
             atoms.get_chemical_formula(), atoms.info["charge"])
    return Path(path)
