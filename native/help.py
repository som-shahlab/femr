import json
import math

import pqdict

with open("all_counts") as f:
    a = json.load(f)

for i, x in a.items():
    print(i, len(x))

infos = pqdict.pqdict(key=lambda a: a[0])

total_counts = {val: count for val, count in a["total_counts"]}
true_counts = {val: count for val, count in a["true_counts"]}
dictionary = {val: count for val, count in a["dictionary"]}
correlations = {(a, b): count for a, b, count in a["correlation"]}

for val, count in total_counts.items():
    true = true_counts.get(val, 0)
    false = count - true
    if true == 0 or false == 0:
        continue

    info = true * math.log(true / count) + false * math.log(false / count)
    infos.additem(val, (info, val, dictionary[val], count, true / count))


final = []
while infos:
    info = infos.popitem()[1]
    bad_keys = []
    for v in infos.values():
        k = (info[1], v[1])
        correlation = correlations.get(k, 0)
        if correlation > 0.95:
            bad_keys.append(v[1])

    for bad_key in bad_keys:
        infos.pop(bad_key)
    final.append(info)

with open("final", "w") as f:
    json.dump(final, f)
