"""usage: python gen_header.py <pendulum|bicycle> <seed> <bits>   -> writes ctrl_params.h next to controller.c"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bits2 as B
from verify_c import setup, train
from export_c import build, header
plant, seed, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
pl, d, X, calib = setup(B.make_pend if plant == "pendulum" else B.make_4d)
layers = build(pl, train(pl, d, seed), b, calib)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ctrl_params.h"), "w").write(header(layers, b, d))
print(f"wrote ctrl_params.h ({plant}, seed {seed}, {b} bits); control u = acc * {float(layers[3]['out_scale']):.12e}, input a_i = round(127*x_i/s_i)")
