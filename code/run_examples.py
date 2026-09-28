"""End-to-end demo: AIMNet2 single point / optimize / vibrations / MD, from SMILES.

Run with the project venv activated:

    source .venv/bin/activate
    python code/run_examples.py

Small molecules and short trajectories, so it finishes in a couple of minutes
on a CPU-only Apple Silicon laptop. Edit the knobs below to scale up.
"""

from __future__ import annotations

import logging
import os

# macOS stability guard (see aimnet_calc.py for rationale).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import aimnet_calc as ac

# ---- knobs -----------------------------------------------------------------
MODEL = "wb97m"      # "b973c" / "2025" also cached; "nse" for open-shell
MD_STEPS = 200
MD_TEMP_K = 300.0


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    section(f"1. SMILES -> 3D  (model={MODEL})")
    # Ethanol. The formal charge is read off the SMILES into atoms.info["charge"].
    ethanol = ac.from_smiles("CCO")
    print(f"{ethanol.get_chemical_formula()}  {len(ethanol)} atoms  "
          f"charge={ethanol.info['charge']}  ff={ethanol.info['forcefield']}")

    section("2. Single point: energy, forces, partial charges, dipole")
    sp = ac.singlepoint(ethanol, model=MODEL)
    print(f"energy      = {sp['energy']:.6f} eV")
    print(f"max |force| = {np.linalg.norm(sp['forces'], axis=1).max():.4f} eV/A")
    print(f"charges     = {np.round(sp['charges'], 3)}")
    print(f"dipole      = {np.round(sp['dipole'], 4)} e*A")

    section("3. Geometry optimization")
    opt = ac.optimize(ethanol, model=MODEL, fmax=0.01, trajectory="ethanol_opt.traj")
    print(f"converged={opt['converged']}  E={opt['energy']:.6f} eV  "
          f"fmax={opt['fmax']:.4f} eV/A")

    section("4. Vibrational frequencies of the relaxed geometry")
    vib = ac.vibrations(ethanol, model=MODEL, name="ethanol_vib")
    freqs = vib["frequencies_cm1"]
    print(f"{vib['nmodes']} modes; highest 5 (cm^-1): "
          f"{np.round(np.real(freqs[-5:]), 1)}")
    n_imag = int((np.real(freqs) < -1.0).sum())
    print(f"imaginary modes: {n_imag} (expect 0 at a true minimum)")

    section("5. Charge awareness: neutral acetic acid vs. acetate anion")
    # The same heavy-atom skeleton at two charges -- AIMNet2 takes the total
    # charge as an input, so this is a genuinely different calculation.
    acid = ac.from_smiles("CC(=O)O")
    anion = ac.from_smiles("CC(=O)[O-]")
    for name, mol in (("acetic acid", acid), ("acetate", anion)):
        r = ac.singlepoint(mol, model=MODEL)
        print(f"{name:12s} charge={mol.info['charge']:+d}  "
              f"E={r['energy']:.4f} eV  sum(q)={r['charges'].sum():+.3f}")

    section(f"6. Short MD on ethanol ({MD_STEPS} steps, {MD_TEMP_K:.0f} K, NVT)")
    md = ac.run_md(ethanol, model=MODEL, T_K=MD_TEMP_K, steps=MD_STEPS,
                   trajectory="ethanol_md.traj", loginterval=50)
    print(f"final energy = {md['final_energy']:.6f} eV -> {md['trajectory']}")

    section("7. XYZ round trip")
    path = ac.smiles_to_xyz("c1ccccc1", "benzene.xyz")
    benzene = ac.from_xyz(path)  # charge=0 default; pass charge= for an ion
    print(f"{path}: {benzene.get_chemical_formula()}, "
          f"E={ac.singlepoint(benzene, model=MODEL)['energy']:.4f} eV")

    print("\nDone.")


if __name__ == "__main__":
    main()
