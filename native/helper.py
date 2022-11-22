import json

with open("final") as f:
    final = json.load(f)

total_info = sum(a[0] for a in final)
cum_info = 0
total_vals = 0
for i, f in enumerate(final):
    cum_info += f[0]
    total_vals += f[-1] * f[-2]
    print(i, cum_info / total_info, total_vals / ((i + 1) * final[0][-2]), f)
