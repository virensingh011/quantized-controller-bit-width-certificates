# How many bits keep a neural network controller safe?

Code and results for the paper *How Many Bits Keep a Neural Network Controller Safe? Error-Margin
Conditions, Sound Certificates and the Seed Lottery in Quantized Neural Feedback Control* (Viren Singh, 2026).

Pure Python (NumPy, SciPy, Matplotlib), CPU only, no GPU. Tested with Python 3.12.

## Install and quick test
```
pip install -r requirements.txt
python smoke_test.py        # about 1 minute; checks that the bounds are sound
```

## What each file does
| File | Purpose |
|---|---|
| `bits2.py` | Plants (pendulum, 4D bicycle), Lyapunov ellipse, numpy MLP training, per-channel weight rounding, difference-interval error bound, weight-norm formula. Running it reproduces Table 1 (bits by method). |
| `bits3.py` | Local certificate (Theorem 4) with cell splitting; interval bounds of the closed loop. |
| `bits4.py` | Change of coordinates (ellipse becomes a ball) for the 4D plant. |
| `bits5.py` | CROWN-style backward bounds (negative result: 4D full-precision network not certified). |
| `bits6.py` | Direct proof, PTQ versus QAT, width study, 4D sampling study. |
| `bits7.py` | Control: fine-tuning without rounding, then PTQ. |
| `mkfig.py` | Makes the figures in `fig/` from the `res*.json` files. |
| `res*.json`, `logs/` | Saved results and console output used in the paper. |

## Reproduce the paper
```
python bits2.py 15      # Table 1 (15 seeds per plant)
python bits3.py 15 6000 # local certificate
python bits6.py 15      # direct proof, PTQ vs QAT, width, 4D sampling
python bits7.py 15      # fine-tuning control
python mkfig.py         # figures
```
Seeds 0 to 14 train the networks; QAT and the fine-tuning control use seed `1000 + seed`.
Runtimes on a laptop CPU are minutes to tens of minutes per script.

## Important notes
* The plants and networks are rebuilt from the descriptions in the paper; they are research scripts, not a library.
* "True bits" are a sampling test (35,000 points in the ellipse), not a proof.
* The four-state plant has no certificate in this code (see the negative result in the paper).
* Results can differ slightly with other NumPy versions because of floating-point order.

## Citation
Add the paper reference and the archive DOI here after publication.

## License
MIT (see `LICENSE`).
