'''
Compare mir_eval against MIREX 2013 chord results
'''

import json
import numpy as np
from mir_eval import chord
from os.path import basename

# Consists of three aligned lists
#   referece_files
#   estimation_files
#   scores
result_json_file = "./data/mirex_results.json"
with open(result_json_file) as fp:
    results = json.load(fp)

VOCABS = ['root', 'majmin', 'majmin-inv', 'sevenths', 'sevenths-inv']
MIREX_VOCABS = [
    "resultsMirexRoot", "resultsMirexMajMin", "resultsMirexMajMinBass",
    "resultsMirexSevenths", "resultsMirexSeventhsBass"]


def filebase(fpath):
    return basename(fpath).split('.')[0]


def get_mir_eval_scores(reference_files, estimation_files):
    '''Computes all mir_eval metrics and returns them in a list with the order
    ['root', 'majmin', 'majmin-inv', 'sevenths', 'sevenths-inv'].
    '''

    scores = dict([(k, list()) for k in VOCABS])

    for ref, est in zip(reference_files, estimation_files):
        single_score = chord.evaluate_file_pair(ref, est, VOCABS)
        for name, score in single_score.items():
            # Skip weights and errors
            if name in scores:
                scores[name].append(score)

    return [100*np.array(scores[name]) for name in VOCABS]

mir_eval_scores = get_mir_eval_scores(results['reference_files'],
                                      results['estimation_files'])


def get_mirex_scores(estimation_files, score_object):
    vocab_scores = dict([(k, list()) for k in MIREX_VOCABS])
    for est_file in estimation_files:
        algo, filename = est_file.split("/")[-2:]
        track_base = filebase(filename)
        for task in MIREX_VOCABS:
            vocab_scores[task].append(score_object[track_base][algo][task])

    return [np.array(vocab_scores[name]) for name in MIREX_VOCABS]


mirex_scores = get_mirex_scores(results['estimation_files'], results['scores'])

errors = np.array([np.abs(a - b)
                   for a, b in zip(mir_eval_scores, mirex_scores)])
