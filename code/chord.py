# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob
import os
import csv
import mir_eval
import numpy as np
import collections

# <codecell>

BASE_DATA_PATH = '../data/chord/'
# What MIREX calls the vocabs
MIREX_VOCABS = ["resultsMirexRoot", "resultsMirexMajMin", "resultsMirexMajMinBass",
                "resultsMirexSevenths", "resultsMirexSeventhsBass"]
# What mir_eval calls the vocabs
VOCABS = ['root', 'majmin', 'majmin_inv', 'sevenths', 'sevenths_inv']
ALG_NAMES = ['CB3', 'CB4', 'CF2', 'KO1', 'KO2', 'NG1', 'NG2', 'NMSD1', 'NMSD2', 'PP3', 'PP4', 'SB8']

# <codecell>

# Convert scores in mirex_scores_raw to the format we want
# These csv files come from http://music-ir.org/mirex/results/2013/ace/MirexChord2009.zip
raw_mirex_dir = os.path.join(BASE_DATA_PATH, 'mirex_scores_raw')
mirex_dir = os.path.join(BASE_DATA_PATH, 'mirex_scores')
for alg in ALG_NAMES:
    if not os.path.exists(os.path.join(mirex_dir, alg)):
        os.makedirs(os.path.join(mirex_dir, alg))
    name_to_scores = collections.defaultdict(list)
    for vocab in MIREX_VOCABS:
        filename = os.path.join(raw_mirex_dir, vocab, alg + '.csv')
        with open(filename, 'r') as f:
            # Skip first two lines
            f.next()
            f.next()
            csv_reader = csv.reader(f)
            for row in csv_reader:
                name_to_scores[row[0]].append(float(row[1])/100.)
    for filename, scores in name_to_scores.items():
        np.savetxt(os.path.join(mirex_dir, alg, filename + '.txt'), scores)

# <codecell>

def process_one_algorithm(algorithm_directory, skip=False):
    ''' Computes mir_eval scores for all output files from one algorithm '''
    reference_dir = os.path.join(BASE_DATA_PATH, 'reference')
    for estimated_chords_file in glob.glob(os.path.join(BASE_DATA_PATH, 'estimated',
                                                        algorithm_directory, '*.lab')):
        est_intervals, est_labels = mir_eval.io.load_labeled_intervals(estimated_chords_file)
        # Skip scores already computed
        if skip and os.path.exists(estimated_chords_file.replace('estimated', 'mir_eval_scores')):
            continue
        # Metrics are computed as the mean across all reference annotations
        for N, reference_chords_file in enumerate(glob.glob(os.path.join(reference_dir, '*',
                                                                         os.path.split(estimated_chords_file)[1]))):
            ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(reference_chords_file)
            scores = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_intervals, est_labels)
            scores = np.array([scores[vocab] for vocab in VOCABS])
        # Compute mean
        scores /= float(N + 1)
        output_path = os.path.split(estimated_chords_file)[0].replace('estimated', 'mir_eval_scores')
        # Make sure output dir exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.savetxt(estimated_chords_file.replace('estimated', 'mir_eval_scores').replace('.lab', '.txt'), scores)

# <codecell>

for alg_name in ALG_NAMES:
    process_one_algorithm(alg_name)

# <codecell>

mir_eval_scores = []
mirex_scores = []
score_glob = glob.glob(os.path.join(BASE_DATA_PATH, 'mir_eval_scores', '*', '*.txt'))
for mir_eval_score_file in score_glob:
    mir_eval_scores.append([np.loadtxt(mir_eval_score_file)])
    mirex_scores.append([np.loadtxt(mir_eval_score_file.replace('mir_eval_scores', 'mirex_scores'))])
mir_eval_scores = np.vstack(mir_eval_scores)
mirex_scores = np.vstack(mirex_scores)

# <codecell>

np.set_printoptions(precision=10, threshold=10000, linewidth=150, suppress=True)

# <codecell>

score_mean = np.mean(np.dstack([np.round(mirex_scores, 4), np.round(mir_eval_scores, 4)]), axis=-1)
score_mean = score_mean + (score_mean == 0)

# <codecell>

diff = np.round(mirex_scores, 3) - np.round(mir_eval_scores, 3)
diff[np.less_equal(np.abs(diff), .0010001)] = 0
print ' & '.join(['{:10s}'.format(key) for key in VOCABS]),
print '\\\\'
print ' & '.join(['{:8.3f}\%'.format(score*100) for score in np.mean(np.abs(diff)/score_mean, axis=0)]),
print '\\\\'

