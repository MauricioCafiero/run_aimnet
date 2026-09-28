# AIMNet2 + ASE on Apple Silicon (small molecules)

Run energy, geometry optimization, vibrational analysis, and MD on small
molecules with the [AIMNet2](https://github.com/isayevlab/aimnetcentral)
foundation models through [ASE](https://wiki.fysik.dtu.dk/ase/), starting from
a **SMILES string or an XYZ file**.

This is the AIMNet2 sibling of the MACE repo: `code/aimnet_calc.py` mirrors
`mace_calc.py` function-for-function, so the same script shape works against
either potential. Scope is deliberately the calculator front-end only — no
fine-tuning, no OOD/trust machinery.

Tested on an Apple Silicon laptop, macOS, **CPU-only**.

## Why AIMNet2 rather than MACE

| | AIMNet2 | MACE-OFF23 |
|---|---|---|
| Elements | 14: H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I — **no metals** | 10: H,C,N,O,F,P,S,Cl,Br,I (OMOL sibling covers Z≤83) |
| Total charge | **an input** — ions are first-class | not modelled |
| Partial charges | **predicted** per atom, plus dipole | not available |
| Hessian | analytic (double backward) | finite differences only |
| Speed (methanol, CPU) | ~4 ms/call after warmup | comparable |

The charge awareness is the reason to reach for it: an anion and its conjugate
acid are genuinely different calculations, not the same one.

## Files

| File | Purpose |
|---|---|
| `code/aimnet_calc.py` | **The module.** SMILES/XYZ → AIMNet2 single point, optimize, vibrations, Hessian, MD. |
| `code/run_examples.py` | End-to-end demo of all of the above on ethanol / acetate / benzene. |

## Environment

A `uv`-managed venv (Python 3.12) lives in `.venv/`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "aimnet[ase,hf]" rdkit matplotlib
source .venv/bin/activate
python code/run_examples.py
```

Verified versions: aimnet 0.2.0, torch 2.14.0, warp-lang 1.17.0, ase 3.29.0,
rdkit 2026.3.6, numpy 2.5.3. The venv is ~1.0 GB (torch dominates).

## Usage

```python
import aimnet_calc as ac

atoms = ac.from_smiles("CCO")          # charge read off the SMILES
ac.optimize(atoms, fmax=0.01)
res = ac.singlepoint(atoms)
res["energy"], res["charges"], res["dipole"]
ac.vibrations(atoms)
ac.run_md(atoms, T_K=300, steps=1000, trajectory="md.traj")

anion = ac.from_smiles("CC(=O)[O-]")   # atoms.info["charge"] == -1, used by the model
xyz   = ac.from_xyz("mol.xyz", charge=-1)   # XYZ carries no charge — pass it here
```

`model=` selects the checkpoint: `"wb97m"` (default), `"b973c"`, `"2025"`,
`"nse"` (spin-aware, the only one that honours `mult`), `"rxn"` (reaction
paths). Ensemble members are `"wb97m-1"` … `"wb97m-3"`; the spread across the
four is a cheap uncertainty estimate.

## Feasibility notes (read this first)

* **It works on Apple Silicon, on the CPU.** torch 2.14 and warp-lang 1.17 both
  ship working arm64 wheels. Verified: optimization converges, vibrations come
  out with zero imaginary modes, MD runs.
* **Timing.** Model load is ~90 s the first time (cold weight load); the *first*
  single point costs ~2 s while warp JIT-compiles its kernels, and every call
  after is **~4 ms** for a 9-atom molecule. So MD and optimization are cheap —
  only pay the warmup once per process.
* **No MPS (Apple GPU).** AIMNet2 is CPU or CUDA; `device="cpu"` is the path here.
* **One stability guard** is set in the module before the torch import:
  `KMP_DUPLICATE_LIB_OK=TRUE` (guards the duplicate-libomp abort). Unlike the
  MACE stack, AIMNet2 does **not** use e3nn, so the `OMP_NUM_THREADS=1` segfault
  workaround is *not* needed — threads are left at the system default.
* **float32 internally.** AIMNet2 has no float64 mode (MACE does). For tight
  vibrations, optimize to a tighter `fmax` rather than expecting float64
  gradients, or use `ac.hessian()` for exact analytic second derivatives.
* **Unsupported elements are a hard error, not an escalation.** There is no
  metal-capable AIMNet2, so `check_elements()` raises and names the offending
  symbols. For metals, use the MACE-OMOL calculator in the sibling repo.

## Model weights

Downloaded on first use to `~/.cache/aimnet/` (~8.8 MB each). Currently cached:
`wb97m`, `b973c`, `2025`. `nse` and `rxn` will download on first use.

Note: MLatom's separate AIMNet2 interface (in the `mlatom_omnip2x` venv) uses a
*different*, JIT-compiled `.jpt` format under `~/.mlatom/models/` and is not
interchangeable with these — and it crashes under current torch. This repo does
not use it.
