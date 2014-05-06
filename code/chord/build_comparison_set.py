"""Mainfile for building a subset of MIREX output data for comparison."""

import argparse
import csv
from collections import OrderedDict
from glob import glob
import json
from os.path import join, basename
from random import shuffle


GROUND_TRUTH = 'Ground-truth'
RESULTS_PREFIX = 'results'
SCORE_COLUMN_IDX = 1


def filebase(fpath):
    return basename(fpath).split('.')[0]


def collect_MIREX_output_data(base_directory):
    """Randomly sample a collection of output data from MIREX.

    Assumes 'base_directory' has the following structure:
    base_directory/
        {AlgorithmCode}/
            {FileBase}.txt
        Ground-truth/   <--- This is specific and required.
            {FileBase}.lab
        ...

    Parameters
    ----------
    base_directory: str
        Path to a directory of MIREX output data (annotations).
    num_pairs: int
        Number of annotation pairs to sample.

    Returns
    -------
    reference_files: list
        List of reference files
    estimation_files: list
        List of algorithmic outputs, corresponding to the references.
    """
    reference_files = glob(join(base_directory, "%s/*.lab" % GROUND_TRUTH))
    estimation_fileset = OrderedDict()
    # Collect estimations and group by filebase.
    for est_file in glob(join(base_directory, "*/*.txt")):
        fbase = filebase(est_file)
        if not fbase in estimation_fileset:
            estimation_fileset[fbase] = []
        estimation_fileset[fbase].append(est_file)

    all_refs, all_ests = [], []
    for ref_file in reference_files:
        for est_file in estimation_fileset[filebase(ref_file)]:
            all_refs.append(ref_file)
            all_ests.append(est_file)

    return all_refs, all_ests


def load_MIREX_results(base_directory):
    """Retrieve the evaluation results for a collection of estimation files.

    Assumes 'base_directory' has the following structure:
    base_directory/
        results{TaskName}/
            {AlgorithmCode}.txt
            {AlgorithmCode}.csv
        ...

    Parameters
    ----------
    base_directory: str
        Path to a directory of MIREX output data (annotations).
    estimation_files: list
        Files to collect from the results.

    Returns
    -------
    results: dunnoyet
        Results corresponding to each file as a dictionary keyed by TaskName.
    """
    results = dict()
    for csv_file in glob(join(base_directory, "%s*/*.csv" % RESULTS_PREFIX)):
        task, filename = csv_file.split("/")[-2:]
        algo = filebase(filename)
        with open(csv_file) as fp:
            reader = csv.reader(fp)
            [reader.next() for n in range(2)]
            for row in reader:
                track_base = row[0]
                if not track_base in results:
                    results[track_base] = dict()
                if not algo in results[track_base]:
                    results[track_base][algo] = dict()
                results[track_base][algo][task] = float(row[SCORE_COLUMN_IDX])

    return results


def main(args):
    ref_files, est_files = collect_MIREX_output_data(args.outputs_directory)
    dataset = dict(
        reference_files=ref_files,
        estimation_files=est_files,
        scores=load_MIREX_results(args.results_directory))
    with open(args.output_file, 'w') as fp:
        json.dump(dataset, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("outputs_directory",
                        metavar="outputs_directory", type=str,
                        help="Directory of chord annotations (outputs).")
    parser.add_argument("results_directory",
                        metavar="results_directory", type=str,
                        help="Directory of algorithm scores.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="Path to save the collected data as JSON.")
    main(parser.parse_args())
