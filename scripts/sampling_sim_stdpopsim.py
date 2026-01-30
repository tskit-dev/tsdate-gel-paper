import argparse
import logging
import os.path
import datetime

import tszip
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import stdpopsim  # currently stdpopsim emits a FutureWarning

argparser = argparse.ArgumentParser(
    description='Run an OOA SLiM simulation on chr17 42:82. Test using sampling_sim_slim.py balanced -n 3'
)
argparser.add_argument(
    'type',
    help='The type of simulation to run',
    choices=['balanced', 'unbalanced'],
)
argparser.add_argument(
    '--slim_path',
    help='path to the SLiM executable. If not given, use msprime for testing purposes',
    default=None
)
argparser.add_argument(
    '--num_samples', '-n',
    type=int,
    help='Number of haploid samples',
    default=60000,  # max diploids in this model at t=0 = 31289 CEU or 12300 YRI
)
args = argparser.parse_args()

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not os.path.isdir(os.path.join(parent_dir, "data")):
    raise ValueError("The `../data` directory for saving files does not exist")

species = stdpopsim.get_species("HomSap")
model = species.get_demographic_model("OutOfAfrica_3G09")
contig = species.get_contig(
    "chr17",
    left=42e6,
    right=82e6,
    genetic_map="HapMapII_GRCh38",
    mutation_rate=model.mutation_rate
)
if args.num_samples % 2 != 0:
    raise ValueError("Number of samples must be even")
diploid_samples = args.num_samples // 2
if args.type == "unbalanced":
    # Proportions from GEL: 95% european, 1% chinese, 4% african
    chb = int(diploid_samples * 0.01)
    ceu = int(diploid_samples * 0.95)
else:
    chb = int(diploid_samples // 3)
    ceu = int(diploid_samples // 3)
    
yri = int(diploid_samples - chb - ceu)
samples = {"YRI": yri, "CHB": chb, "CEU": ceu}

print(f"Simulating {samples} diploid samples")

params = {}
if args.slim_path:
    dfe = species.get_dfe("Mixed_K23")
    exons = species.get_annotations("ensembl_havana_104_exons")
    exon_intervals = exons.get_chromosome_annotations("chr17")
    contig.add_dfe(intervals=exon_intervals, DFE=dfe)
    engine = stdpopsim.get_engine("slim")
    params["slim_path"] = args.slim_path
    params['slim_burn_in'] = 10
else:
    engine = stdpopsim.get_engine("msprime")
    logging.warning("USING msprime (FOR TESTING ONLY)")

start = datetime.datetime.now()
ts = engine.simulate(model, contig, samples, seed=123, **params)
print(
    f"Simulated {ts.num_trees} trees ({ts.num_sites} sites / {ts.num_samples} samples) "
    f"in {datetime.datetime.now() - start} secs"
)
if ts.num_samples != args.num_samples:
    samples = {p.metadata['name']: len(ts.samples(population=p.id)) for p in ts.populations()}
    logging.warning(
        f"Created {samples} (ts.num_samples) != {args.num_samples} requested samples"
    )
tszip.compress(
    ts,
    os.path.join(parent_dir, "data", f"sampling_sim_{args.type}+{ts.num_samples}.tsz"))
