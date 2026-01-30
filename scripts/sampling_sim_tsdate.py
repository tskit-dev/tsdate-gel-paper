import argparse
import json
import os.path
import re

import tskit
import tsdate
import tszip

argparser = argparse.ArgumentParser(description='Run tsdate on tsinferred output in ../data')
argparser.add_argument(
    'type',
    help='The type of simulation to date',
    choices=['balanced', 'unbalanced'],
)
argparser.add_argument(
    '--num_samples', '-n',
    type=int,
    help='Number of diploid samples',
    default=None,  # default involves checking if there are any existing files
)
args = argparser.parse_args()


print(f"Using version {tsdate.__version__} of tsdate", flush=True)

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(parent_dir, "data")
files = {
    m.group(1):fn
    for fn in os.listdir(data_dir)
    if (m := re.match(fr"sampling_sim_{args.type}\+(\d+).inferred.tsz", fn))
}
n = args.num_samples
if n is None:
    if len(files) == 1:
        n = int(list(files.keys())[0])
    else:
        raise ValueError(
            f"Multiple files (n={list(files.keys())}, please specify which via --num_samples"
        )
input = f"sampling_sim_{args.type}+{n}.inferred.tsz"
print("Loading", input)
its = tszip.load(os.path.join(data_dir, input))
for p in reversed(its.provenances()):
    if json.loads(p.record)['parameters']['command'] == "sim_mutations":
        r = json.loads(p.record)['parameters']["rate"]
        try:
            mu = r['rate']['__ndarray__']
            assert len(mu) == 1
            mu = mu[0]
        except TypeError:
            mu = r
        break

dts = tsdate.date(tsdate.preprocess_ts(its), mutation_rate=mu, progress=True)
tszip.compress(dts, os.path.join(data_dir, f"sampling_sim_{args.type}+{n}.dated.tsz"))
