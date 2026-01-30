# Select the reliable portions of chromosomes, e.g. excluding the centromere
# (byt default we take the chr20 regions used in the main GEL inference)
import argparse
import os

import tsdate
import tszip

PHLASH_MUTATION_RATE = 1.29e-8

def main(args):
    if len(args.chromosomes) != len(args.regions):
        raise ValueError("The number of chromosomes and regions must match")
    for i, (chrom, region) in enumerate(zip(args.chromosomes, args.regions)):
        fn = args.input_name_format.format(chrom=chrom)
        print("Dating chromosome", chrom, "region", region, "from", fn)
        ts = tszip.load(fn)
        if region == "all":
            region = [0, ts.sequence_length]
            for i, tree in enumerate([ts.first(), ts.last()]):
                if tree.num_edges == 0:
                    # if first, use the RH interval, if last, use the LH interval
                    region[i] = tree.interval[1-i]
            print(
                f"Using all of {chrom} "
                f"({ts.sequence_length} bp, of which data exists between {region})"
            )
        else:
            region = [int(x) for x in region.strip().split("-")]
            if len(region) != 2:
                raise ValueError("Must have start and end of region separated by a hyphen")
            ts = ts.keep_intervals([region])
        ts = tsdate.preprocess_ts(ts)
        dts = tsdate.date(ts, mutation_rate=PHLASH_MUTATION_RATE, singletons_phased=False, progress=True)
        try:
            outfile = args.outfiles[i]
        except (IndexError, TypeError):
            dir = os.path.dirname(args.input_name_format)
            outfile = f"{dir}/{chrom}_{region[0]}-{region[1]}.tsz"
        print("Saving to", outfile)
        tszip.compress(dts, outfile)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='Redate selected portions of the 1000G inferred tree sequences')
    argparser.add_argument(
        '--input_name_format', "-i",
        help=(
            'The inferred tree sequence filename format, with {chrom} as a placeholder'
        ),
        default="data/tgp/all-{chrom}-filterNton23-truncate-0-0-0-mm0-post-processed-simplified-SDN-singletons-dated-metadata.trees.tsz"
    )
    argparser.add_argument(
        '--chromosomes', "-c",
        type=str,
        nargs='+',
        help='The chromosomes to use',
        default=["chr20p", "chr20q"]
    )
    argparser.add_argument(
        '--regions', "-r",
        type=str,
        nargs='+',
        help='The regions of each chromosome to use, or "all" for all',
        default=["1-28100000", "31078820-64318092"]
    )
    argparser.add_argument(
        '--outfiles', "-o",
        type=str,
        nargs='+',
        help='Optional names for the output files',
        default=None,
    )
    args = argparser.parse_args()
    main(args)

