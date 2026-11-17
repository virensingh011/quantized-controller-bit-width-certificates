import json, bits2 as B, bits12_int as I
name, rows = I.study(B.make_4d, 15); I.summarize(name, rows); json.dump({name: rows}, open("res12_int_4d.json","w"))
