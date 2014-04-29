"""Mainfile for building a subset of MIREX output data for comparison."""

import argparse
import csv
from glob import glob
import json
from os.path import join, basename
from random import shuffle


GROUND_TRUTH = 'Ground-truth'
RESULTS_PREFIX = 'results'
SCORE_COLUMN_IDX = 1


def filebase(fpath):
    return basename(fpath).split('.')[0]


def sample_MIREX_output_data(base_directory, num_pairs=10):
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
    reference_annotations = glob(join(base_directory,
                                      "%s/*.lab" % GROUND_TRUTH))

    estimated_annotations = dict()
    # Collect estimations and group by filebase.
    for est_file in glob(join(base_directory, "*/*.txt")):
        fbase = filebase(est_file)
        if not fbase in estimated_annotations:
            estimated_annotations[fbase] = []
        estimated_annotations[fbase].append(est_file)

    # Randomize estimated files and only keep one.
    for fbase in estimated_annotations:
        shuffle(estimated_annotations[fbase])
        estimated_annotations[fbase] = estimated_annotations[fbase][0]

    # Shuffle and slice reference annotations.
    shuffle(reference_annotations)
    ref_files = reference_annotations[:num_pairs]
    est_files = [estimated_annotations[filebase(ref_file)]
                 for ref_file in ref_files]
    return ref_files, est_files


def retrieve_MIREX_results(base_directory, estimation_files):
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
    results: list of dicts
        Results corresponding to each file as a dictionary keyed by TaskName.
    """
    results = []
    for est_file in estimation_files:
        algo, filename = est_file.split("/")[-2:]
        fbase = filebase(filename)
        res_fmt = join(base_directory, "%s*/%s.csv" % (RESULTS_PREFIX, algo))
        results.append(dict())
        for res_file in glob(res_fmt):
            taskname = res_file.split("/")[-2].replace(RESULTS_PREFIX, '')
            score = None
            with open(res_file) as fp:
                reader = csv.reader(fp)
                for row in reader:
                    if row[0] == fbase:
                        score = float(row[SCORE_COLUMN_IDX])
                        break
            results[-1][taskname] = score

    return results


def main(args):
    ref_files, est_files = sample_MIREX_output_data(
        args.outputs_directory, args.num_pairs)
    dataset = dict(
        reference_files=ref_files,
        estimation_files=est_files,
        scores=retrieve_MIREX_results(args.results_directory, est_files))
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
    parser.add_argument("--num_pairs", default=10,
                        metavar="--num_pairs", type=int,
                        help="Directory of algorithm scores.")
    main(parser.parse_args())
