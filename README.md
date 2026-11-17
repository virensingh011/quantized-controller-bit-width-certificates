# Quantized neural feedback control: how much rounding can a certified invariant ellipsoid survive?

Code and results for *Quantized Neural Feedback Control: How Much Rounding Can a Certified Invariant Ellipsoid
Survive?* (Viren Singh, 2026). Pure Python (NumPy, SciPy, Matplotlib), CPU only, no GPU. Tested with Python 3.12.

## Install and quick test
```
pip install -r requirements.txt
python smoke_test.py        # about 1 minute; checks that the pendulum bounds are sound
python test9.py 0           # about 1 minute; soundness tests of the four-state verifier (random boxes)
```

## What each file does
| File | Purpose |
|---|---|
| `bits2.py` | Plants (pendulum, 4D bicycle), Lyapunov ellipse, numpy MLP training, per-channel weight rounding, difference-interval error bound, weight-norm formula. Reproduces the "bits by method" rows. |
| `bits3.py` | Local certificate (Theorem 3 of the ESL paper) with cell splitting; interval bounds of the closed loop (pendulum). |
| `bits4.py`, `bits5.py` | Earlier four-state attempts (coordinate change; CROWN-style bounds with interval fallback): the negative result for the interval verifier. |
| `bits6.py` | Direct proof, PTQ versus QAT, width study, 4D sampling study. |
| `bits7.py`, `bits8.py` | Fine-tuning control; pendulum at 5 bits. |
| `bits9.py` | **Verifier for plants affine in the control** (Appendix A): full CROWN bounds, saturation relaxation, vertex and ball bounds, box splitting. |
| `bits10.py` | **Four-state certificates**: full-precision network, direct proofs and local certificates, 15 seeds (`python bits10.py <seed>`). |
| `bits11.py` | Verifier ablation (`python bits11.py ablation 0 1 6`) and margin table (`python bits11.py margin`). |
| `bits11_budget.py`, `bits12_budget.py` | Local-certificate width versus box budget (150,000 boxes, 1e-6 margin). |
| `bits12_int.py`, `run12_4d.py` | Integer-arithmetic emulation of the quantized controllers (both plants / bicycle only). |
| `embedded/` | **Integer-only C controller** (`controller.c`), exporter, bit-exact check against a NumPy fixed-point model, size table, ARM build script. See `embedded/README.md`. |
| `test9.py` | Soundness tests of `bits9.py`: every bound must contain what sampling finds. |
| `test_cor3.py`, `test_di.py` | Corollary 3 check; third plant (double integrator). |
| `results_4d_summary.csv` | Per-seed four-state results (contraction, margin, sampling bits, direct and local certificate bits, box counts). |
| `res*.json`, `logs/`, `log10_*`, `log11_*` | Saved results and console output. |

## Reproduce
```
python bits2.py 15 ; python bits3.py 15 6000 ; python bits6.py 15 ; python bits7.py 15 ; python bits8.py 15
./run_4d_certificates.sh                  # four-state certificates (bits9/bits10)
python bits11.py ablation 0 1 6 ; python bits11.py margin ; python bits12_budget.py 0 1 6 10 14 ; python bits12_int.py 15
```
Seeds 0 to 14 train the networks; QAT and the fine-tuning control use seed `1000 + seed`.

## Four-state results (15 seeds)
Budget 40,000 boxes, relative safety margin 1e-6 on every bound. Full-precision network certified for all 15 seeds
(15.7k to 24.4k boxes, at most 8 s each). Direct proof: mean 4.8 bits (4 bits for 7 seeds, 5 for five, 6 for two, 7 for
one). Local certificate: mean 9.9 bits (9 for three seeds, 10 for eleven, 11 for one).
* **Box budget** (`bits12_budget.py`): with 150,000 boxes the local certificate needs 8, 8, 9, 8, 8 bits for seeds
  0, 1, 6, 10, 14 (9, 11, 10, 9, 9 at 40,000 boxes). The search started at 6 bits.
* **Ablation** (`bits11.py ablation 0 1 6`): interval intermediate bounds or no saturation relaxation: not certified;
  no vertex bound: not certified for 2 of 3; no ball bound: certified with 8 to 17 percent more boxes; no interval
  fallback: unchanged.
* **Integer emulation** (`bits12_int.py`, sampling test, not a proof): 8-bit or 12-bit activations and integer biases
  leave the pendulum unchanged (3.80 first pass, 3.93 true bits); the bicycle moves from 4.20 / 4.60 (float) to
  4.27 / 4.53 (8-bit) and 4.20 / 4.47 (12-bit); all networks are safe from 6 bits.
* **Integer-only C controller** (`embedded/verify_c.py`): the C code equals a NumPy fixed-point model bit for bit on 6,000
  points (3, 4, 5, 8 bits, both plants); closed-loop sampling results equal the integer emulation above; constant data
  at 4 bits: 1,944 B (pendulum) and 1,976 B (bicycle) against 8,964 and 9,220 B in float32. Code size and timing on a
  Cortex-M were not measured (`embedded/build_arm.sh` does it with `arm-none-eabi-gcc`).
* `old_nomargin_res10/` holds the same runs without the safety margin (identical per-seed widths).

## Important notes
* **`bits9.py`, `bits10.py`, `bits11.py` were re-implemented on 2026-11-15 from the description of the verifier after the
  original files were lost.** The four-state certificate numbers above (and in the paper) come from this re-implementation.
  The 15 four-state networks retrain exactly as before (same contraction and sampling "true bits" as the saved logs).
* The verifier uses float64 without outward rounding (a 1e-6 relative safety margin is applied to every bound), so a "proof" is exact only up to floating-point rounding.
* "True bits" are a sampling test (35,000 points in the ellipse), not a proof.
* The plants and networks are rebuilt for this study; these are research scripts, not a library.
* Results can differ slightly with other NumPy versions because of floating-point order.

## Citation
Archived at Zenodo (release `v2.0-esl`): [10.5281/zenodo.24269026](https://doi.org/10.5281/zenodo.24269026).

V. Singh, *Quantized Neural Feedback Control: How Much Rounding Can a Certified Invariant Ellipsoid Survive?*, 2026.

## License
MIT (see `LICENSE`).
