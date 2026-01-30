import argparse
import os.path
import re

import tskit
import tsinfer
import tszip

argparser = argparse.ArgumentParser(
    description='Run tsinfer on OOA simulation output in sampling_sim*.tsz')
argparser.add_argument(
    'type',
    help='The type of simulation to run',
    choices=['balanced', 'unbalanced'],
)
argparser.add_argument(
    '--num-threads',
    type=int,
    default=40,
    help='The number of threads to use for inference',
)
argparser.add_argument(
    '--num_samples', '-n',
    type=int,
    help='Number of diploid samples',
    default=None,  # default involves checking if there are any existing files
)
args = argparser.parse_args()

print(f"Using version {tsinfer.__version__} of tsinfer", flush=True)

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(parent_dir, "data")

files = {
    m.group(1):fn
    for fn in os.listdir(data_dir)
    if (m := re.match(fr"sampling_sim_{args.type}\+(\d+).tsz", fn))
}
n = args.num_samples
if n is None:
    if len(files) == 1:
        n = int(list(files.keys())[0])
    else:
        raise ValueError(
            f"Multiple files (n={list(files.keys())}, please specify which via --num_samples"
        )
fn = f"sampling_sim_{args.type}+{n}.tsz"
print(f"Loading {fn}")
ts = tszip.load(os.path.join(data_dir, fn))
tables = ts.dump_tables()
# change individual metadata (e.g. placed by SLiM) to JSON, which is required by tsinfer
tables.individuals.drop_metadata()
tables.individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
tables.individuals.packset_metadata(
    [tables.individuals.metadata_schema.validate_and_encode_row(i.metadata or {})
     for i in ts.individuals()])

its = tsinfer.infer(
    tsinfer.SampleData.from_tree_sequence(tables.tree_sequence()),
    progress_monitor=True,
    num_threads=args.num_threads)

tszip.compress(its, os.path.join(data_dir, f"sampling_sim_{args.type}+{n}.inferred.tsz"))
