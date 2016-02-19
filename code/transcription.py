# CREATED: 2/17/16 3:09 PM by Justin Salamon <justin.salamon@nyu.edu>

'''
Compare transcription results reported in MIREX 2015 for the Note Tracking
subtask of multi-f0 challenge against those produce by the mir_eval
transcription module
'''

import numpy as np
import mir_eval
import glob, os
import pandas as pd
import sys


def load_mirex_scores(alg):
    '''
    Load mirex scores for a specific algorithm

    Parameters
    ----------
    alg : str
        Algorithm name

    Returns
    -------
    mirex_scores : np.ndarray, shape=(10,6)
        The scores. There are 10 tracks, and 6 evaluation metrics: Precision,
        Recall, F-measure, Precision_no_offset, Recall_no_offset,
        F-measure_no_offset
    '''

    mirex_scores = []

    mirex_results_path = \
        'data/transcription/mirex2015_su/mirex_results/' \
        '%s/%s.results.csv' % (alg, alg)
    mirex_nooffset_results_path = \
        'data/transcription/mirex2015_su/mirex_results/' \
        '%s/%s.onsetOnly.results.csv' % (alg, alg)

    mirex_results = pd.read_csv(mirex_results_path)
    mirex_results = mirex_results.fillna(0)
    precision = mirex_results['Precision'].tolist()[:10]
    recall = mirex_results['Precision'].tolist()[:10]
    f1 = mirex_results['Ave. F-measure'].tolist()[:10]

    mirex_nooffset_results = pd.read_csv(mirex_nooffset_results_path)
    mirex_nooffset_results = mirex_nooffset_results.fillna(0)
    precision_nooffset = mirex_nooffset_results['Precision'].tolist()[:10]
    recall_nooffset = mirex_nooffset_results['Precision'].tolist()[:10]
    f1_nooffset = mirex_nooffset_results['Ave. F-measure'].tolist()[:10]

    mirex_scores.append(precision)
    mirex_scores.append(recall)
    mirex_scores.append(f1)
    mirex_scores.append(precision_nooffset)
    mirex_scores.append(recall_nooffset)
    mirex_scores.append(f1_nooffset)
    mirex_scores = np.asarray(mirex_scores)
    mirex_scores = mirex_scores.T

    filenames = mirex_results['Filename'].tolist()[:10]
    filenames = np.asarray(filenames)
    filenames_nooffset = mirex_nooffset_results['Filename'].tolist()[:10]
    filenames_nooffset = np.asarray(filenames_nooffset)

    assert (filenames==filenames_nooffset).all()
    mirex_filenames = [fn.replace("_tutti.mid", "") for fn in filenames]
    mirex_filenames = np.asarray(mirex_filenames)

    return mirex_scores, mirex_filenames


def compute_mireval_scores(alg):
    '''
    Compute mireval evaluation scores for a specific algorithm

    Parameters
    ----------
    alg : str
        Algorithm name

    Returns
    -------
    mir_eval_scores: np.ndarray, shape=(10,6)
        The scores. There are 10 tracks, and 6 evaluation metrics: Precision,
        Recall, F-measure, Precision_no_offset, Recall_no_offset,
        F-measure_no_offset
    '''

    # ref_folder = 'data/transcription/mirex2015_su/reference/gt_Note'
    ref_folder = 'data/transcription/mirex2015_su/reference/gt_textfrommidi'
    ref_tracks = glob.glob(os.path.join(ref_folder, "*.txt"))

    est_folder = 'data/transcription/mirex2015_su/estimate/%s/' % alg
    est_tracks = glob.glob(os.path.join(est_folder, "*.txt"))

    mir_eval_scores = []
    mir_eval_filenames = []

    # Collect mir_eval scores
    for ref_path, est_path in zip(ref_tracks, est_tracks):

        assert os.path.basename(ref_path).replace("_tutti.mid.txt","") == \
               os.path.basename(est_path).replace(".txt","")
        mir_eval_filenames.append(os.path.basename(est_path).replace(".txt",""))

        ref_int, ref_pitch = mir_eval.io.load_valued_intervals(ref_path,
                                                               delimiter=',')
        est_int, est_pitch = mir_eval.io.load_valued_intervals(est_path)
        scores = mir_eval.transcription.evaluate(ref_int, ref_pitch, est_int,
                                                 est_pitch)

        score_list = []
        for metric in scores:
            score_list.append(scores[metric])

        mir_eval_scores.append(score_list)

    return np.asarray(mir_eval_scores), np.asarray(mir_eval_filenames)


def compare_scores(mirex_scores, mir_eval_scores):
    '''
    Compare mirex scores to mir_eval scores.

    Parameters
    ----------
    mirex_results : np.ndarray, shape=(n, 6)
        Array with mirex eval results, each row represents one song, and the
        six result columns contain the following metrics: Precision, Recall,
        F-measure, Precision_no_offset, Recall_no_offset, F-measure_no_offset
    mireval_results: np.ndarray, shape=(n, 6)
        Array with mir_eval eval results, each row represents one song, and the
        six result columns contain the following metrics: Precision, Recall,
        F-measure, Precision_no_offset, Recall_no_offset, F-measure_no_offset
    :return:
    '''

    assert mirex_scores.shape == mir_eval_scores.shape

    # print(mirex_scores)
    # print(" ")
    # print(mir_eval_scores)
    # print(" ")

    print(" ")
    print("P\tR\tF\tP_nooff\tR_nooff\tF_nooff")
    print('----------------------------------------------')
    diff = mir_eval_scores - mirex_scores
    for line in diff:
        for val in line:
            sys.stdout.write("%.3f\t" % val)
        sys.stdout.write("\n")
    print('----------------------------------------------')

    means = diff.mean(0)
    for m in means:
        sys.stdout.write("%.3f\t" % m)
    sys.stdout.write("(mean diff)\n")

    abs_means = np.abs(diff).mean(0)
    for m in abs_means:
        sys.stdout.write("%.3f\t" % m)
    sys.stdout.write("(mean abs diff)\n")



    all_close_3 = np.allclose(mirex_scores, mir_eval_scores, atol=1e-3)
    all_close_2 = np.allclose(mirex_scores, mir_eval_scores, atol=1e-2)
    all_close_1 = np.allclose(mirex_scores, mir_eval_scores, atol=1e-1)
    print("All close 1e-3: {}".format(all_close_3))
    print("All close 1e-2: {}".format(all_close_2))
    print("All close 1e-1: {}".format(all_close_1))
    print(" ")
    # print(" ")

    print("mireval scores:")
    for line in mir_eval_scores:
        for val in line:
            sys.stdout.write("%.3f\t" % val)
        sys.stdout.write("\n")
    print(" ")

    print("mirex scores:")
    for line in mirex_scores:
        for val in line:
            sys.stdout.write("%.3f\t" % val)
        sys.stdout.write("\n")
    print(" ")


def compare_alg_results(alg):
    '''
    Compare mirex and mir_eval results for a specific algorithm

    Parameters
    ----------
    alg : str
        Name of algorithms (BW2, BW2, CB1, etc.)
    :return:
    '''

    mirex_scores, mirex_filenames = load_mirex_scores(alg)
    mir_eval_scores, mir_eval_filenames = compute_mireval_scores(alg)

    assert (mirex_filenames==mir_eval_filenames).all()

    compare_scores(mirex_scores, mir_eval_scores)


def compare_all_alg_results():

    results_path = 'data/transcription/mirex2015_su/mirex_results'

    algs = glob.glob(os.path.join(results_path, "*"))

    for alg in algs:

        if os.path.isdir(alg):

            alg_name = os.path.basename(alg)
            print("Algorithm: %s" % alg_name)
            compare_alg_results(alg_name)


if __name__ == "__main__":

    # compare_all_alg_results()
    compare_alg_results('BW2')