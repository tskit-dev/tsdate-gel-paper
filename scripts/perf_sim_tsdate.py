import argparse
import json
import subprocess
import sys
import os.path
import tempfile
from datetime import datetime

import numpy as np
import tskit
import tszip
from tqdm.auto import tqdm


def simplify_removing_mutations_above_roots(ts, *args, **kwargs):
    # Make sure we have the mutation parents correct.
    # Also store the original node IDs in the node metadata
    tables = ts.dump_tables()
    tables.compute_mutation_parents()
    node_map = tables.simplify(*args, **kwargs)
    rev_map = np.zeros_like(node_map, shape=tables.nodes.num_rows)
    kept = node_map != tskit.NULL
    rev_map[node_map[kept]] = np.arange(len(node_map))[kept]
    if len(tables.nodes.metadata) == 0:
        # Save the original node ID in the metadata, so we can compare later
        tables.nodes.metadata_schema = tskit.MetadataSchema({
                    "codec": "struct",
                    "type": "object",
                    "properties": {
                        "mn": {"type": "number", "binaryFormat": "f", "default": float("NaN")},
                        "vr": {"type": "number", "binaryFormat": "f", "default": float("NaN")},
                        "original_id": {"type": "number", "binaryFormat": "i", "default": -1}
                    },
        })
        tables.nodes.packset_metadata([
            tables.nodes.metadata_schema.encode_row({"original_id": i}) for i in rev_map])
    else:
        # Check that we have already set the original node IDs
        if np.any(ts.nodes_metadata["original_id"] < 0):
            raise ValueError("The original node IDs have not been set in the metadata.")
    ts = tables.tree_sequence()
    keep_muts = np.ones(ts.num_mutations, dtype=bool)
    deleted_mut_parents = set()
    for tree in ts.trees():
        for site in tree.sites():
            root = tree.root
            root_muts = [mut for mut in site.mutations if mut.node == root]
            if len(root_muts) > 0:
                # Take the last mutation above the root
                anc_state = root_muts[-1].derived_state
                deleted_mut_parents.add(root_muts[-1].id)
                keep_muts[[mut.id for mut in root_muts]] = False
                tables.sites[site.id] = tables.sites[site.id].replace(ancestral_state=anc_state)
    # Some non-root mutations may have a parent that we intend to delete
    # We can safely set the parent of these mutations to NULL, as we have
    # now set the ancestral state to the state of that parent
    deleted_mut_parents = np.array(list(deleted_mut_parents))
    mut_parents = tables.mutations.parent
    mut_parents[np.isin(mut_parents, deleted_mut_parents)] = tskit.NULL
    tables.mutations.parent = mut_parents
    tables.mutations.keep_rows(keep_muts)
    return tables.tree_sequence()

def main(args):
    rng = np.random.default_rng(args.random_seed)
    big_ts = tszip.load(args.input)
    if args.initial_num_samples is not None:
        print("Reducing the input tree sequence to", args.initial_num_samples, "samples")
        if args.initial_num_samples > big_ts.num_samples:
            raise ValueError("initial_num_samples cannot be greater than the number of samples in the input")
        rand_samples = np.arange(args.initial_num_samples)
        np.random.shuffle(rand_samples)
        big_ts = simplify_removing_mutations_above_roots(big_ts, rand_samples)

    mu = None
    for provenance in reversed(big_ts.provenances()):
        rec = json.loads(provenance.record)
        if rec["parameters"]["command"] == "sim_mutations":
            mu = rec["parameters"]["rate"]
            print("Using a mutation rate of", mu)
    if mu is None:
        raise ValueError("Could not find mutation rate in provenance")
    num_samples = np.array(args.size_fractions) * big_ts.num_samples
    num_samples = np.rint(num_samples).astype(int)
    if min(num_samples) < 10:
        raise ValueError(
            "Size reductions must be set so that the minimum sample size is at least 10")
    if max(num_samples) > big_ts.num_samples:
        raise ValueError(
            "Size reductions cannot be greater than one")
    print(
        f"The minimum sample size, used to define variable sites, is {min(num_samples)}")
    rand_samples = np.arange(big_ts.num_samples)
    rng.shuffle(rand_samples)

    # get the version of tsdate that is installed using the cli
    cmd = ["python", "-m", "tsdate", "-V"]
    version = subprocess.run(cmd, capture_output=True, text=True).stdout.split()[-1]
    short_version = version.split("+")[0]

    cmd = ["python", "-m", "tsdate", "date", "-m", f"{mu}"]
    if args.use_usr_bin_time:
        if sys.platform == "darwin":
            cmd = ["/usr/bin/time", "-l"] + cmd
        else:
            cmd = ["/usr/bin/time", "-f", "%M  maximum resident set size"] + cmd
    method = ""
    if args.population_size is not None:
        try:
            population_size = float(args.population_size)
        except ValueError:
            if args.population_size.lower() == "true":
                population_size = big_ts.diversity() / (4 * mu)
                print("Using population size from data (π/4µ):", population_size)
            else:
                raise ValueError("population_size must be a number or 'True'")
        method = "inside_outside"
        cmd.extend(["--method", method, "--population_size", f"{population_size}"])

    with tempfile.TemporaryDirectory() as tmpdir:
        for s in tqdm(num_samples, total=len(num_samples)):
            # Take the first N. Also remove any mutations above the
            # local roots, and change the ancestral state accordingly
            print("Dating a subset of", s, "out of", big_ts.num_samples, "samples")
            ts = simplify_removing_mutations_above_roots(big_ts, rand_samples[np.arange(0, s)])

            infile = f"simulated_chrom_17-{s}_{args.random_seed}"
            outfile = (
                f"simulated_chrom_17-{s}+{args.ep_iterations}_{args.random_seed}" +
                f".tsdate{short_version}{method}"
            )
            ts.dump(f"{tmpdir}/{infile}.trees")
            print(f"Running tsdate {version} on {s} samples")
            loop_cmd = cmd.copy()
            if args.ep_iterations is not None:
                loop_cmd.extend(["--max-iterations", f"{args.ep_iterations}"])
            loop_cmd.extend([f"{tmpdir}/{infile}.trees", f"{tmpdir}/{outfile}.trees"])
            print(f"Running command: {' '.join(loop_cmd)}")
            start = datetime.now()
            # capture the output of the /usr/bin/time command
            result = subprocess.run(loop_cmd, capture_output=True, text=True)
            if not os.path.exists(f"{tmpdir}/{outfile}.trees"):
                raise RuntimeError("Output file not created")
            time_taken = datetime.now() - start
            if result.returncode != 0:
                print("Error running tsdate:", result.stderr)
                continue
            dts = tszip.load(f"{tmpdir}/{outfile}.trees")
            rec = json.loads(dts.provenance(-1).record)
            print(f"Time taken: {time_taken} (recorded in provenance={rec['resources']['elapsed_time']})")
            if args.use_usr_bin_time:
                memory_bytes = "Unknown"
                for line in result.stderr.strip().split("\n"):
                    parts = line.rsplit("  ", 1)
                    if parts[-1] == "maximum resident set size":
                        # This is the memory usage in KB
                        memory = int(parts[0])
                        memory_bytes = memory if sys.platform == "darwin" else memory * 1024
                        break
                print(f"Mem recorded in provenance={rec['resources']['max_memory']}) vs actual={memory_bytes} bytes")
                tables = dts.dump_tables()
                tables.metadata_schema = tskit.MetadataSchema.permissive_json()
                tables.metadata = {"mem from /usr/bin/time": memory_bytes}
                dts = tables.tree_sequence()
            tszip.compress(dts, f"data/{outfile}.tsz")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Run tsdate performance tests on a large simulation')
    argparser.add_argument(
        'input',
        help=(
            'The simulation input file, with true times. Suggested files to put in the data directory '
            'can be found at https://zenodo.org/records/7702392 (Quebec simulations)'
        ),
        nargs='?',
        default="data/simulated_chrom_17.ts.tsz",
    )
    argparser.add_argument(
        '--size_fractions', "-f",
        type=float,
        nargs='+',
        help='Fractions of the total sample size. The smallest is taken as the minimum sample size',
        default=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    )
    argparser.add_argument(
        '--initial_num_samples', "-n",
        type=int,
        help='The initial number of samples: the input ts will be reduced to this size',
        default=None
    )
    argparser.add_argument(
        '--ep-iterations', '-i',
        type=int,
        help='Number of Expectation Propagation iterations',
        default=None, 
    )
    argparser.add_argument(
        '--population_size',
        type=str,
        help=(
            'If a number or "True"/"true", the old inside-outside method will be used. '
            'If True, the population size will be estimated from the data using π/4µ.'
        ),
        default=None, 
    )
    argparser.add_argument(
        '--random-seed', '-s',
        help='The random seed used to select samples from the larger population',
        type=int,
        default=123, 
    )
    argparser.add_argument(
        '--use-usr-bin-time', '-t',
        help='Use the /usr/bin/time command to measure memory usage',
        action='store_true',
    )
    args = argparser.parse_args()
    main(args)

