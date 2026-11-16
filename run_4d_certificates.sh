#!/bin/sh
# four-state plant: full-precision certificate, direct proofs and local certificates for the 15 seeds
# (about 1-5 minutes per seed on one CPU core; bits10.py is resumable, results go to res10_seed<k>.json)
for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do python bits10.py $s >> log10_seed$s.txt 2>&1; done
