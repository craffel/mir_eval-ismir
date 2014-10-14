# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import urllib
import glob
import os
import csv
import mir_eval
import numpy as np

# <codecell>

BASE_DATA_PATH = '../data/melody'
METRIC_KEYS = ['Overall Accuracy', 'Raw Pitch Accuracy', 'Raw Chroma Accuracy', 'Voicing Recall Rate', 'Voicing False-Alarm Rate']
ALG_NAMES = ['SG2']

# <codecell>

RESULT_URL = "http://nema.lis.illinois.edu/nema_out/mirex2011/results/ame/adc04/{}/per_track_results.csv"
csv_dir = os.path.join(BASE_DATA_PATH, 'mirex_scores_raw')
try:
    os.makedirs(csv_dir)
except OSError:
    pass
for alg in ALG_NAMES:
    file_url = RESULT_URL.format(alg)
    urllib.urlretrieve(file_url, os.path.join(csv_dir, alg + '.csv'))

# <codecell>

for filename in glob.glob(os.path.join(csv_dir, '*.csv')):
    output_list = []
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            # Skip the label row
            if row[0] != '*Fold':
                # Construct the filename/output path
                output_filename = 'sg2melodytranscript' + row[1].replace('_', '') + '.txt'
                output_path = filename.replace('mirex_scores_raw', 'mirex_scores')
                output_path = os.path.join(os.path.split(output_path)[0], os.path.split(output_path)[1].replace('.csv', ''))
                # Make sure output dir exists
                try:
                    os.makedirs(output_path)
                except OSError:
                    pass
                np.savetxt(os.path.join(output_path, output_filename), [float(x) for x in row[2:]])

# <codecell>

def process_one_algorithm(algorithm_directory, skip=False):
    ''' Computes mir_eval scores for all output files from one algorithm '''
    for estimated_melody_file in glob.glob(os.path.join(BASE_DATA_PATH, 'estimated',
                                                        algorithm_directory, '*.txt')):
        est_time, est_freq = mir_eval.io.load_time_series(estimated_melody_file)
        # Skip scores already computed
        if skip and os.path.exists(estimated_melody_file.replace('estimated', 'mir_eval_scores')):
            continue
        scores = np.zeros(len(METRIC_KEYS))
        # Metrics are computed as the mean across all reference annotations
        for N, reference_melody_file in enumerate(glob.glob(os.path.join(BASE_DATA_PATH, 'reference', '*',
                                                                         os.path.split(estimated_melody_file)[1]))):
            ref_time, ref_freq = mir_eval.io.load_time_series(reference_melody_file)
            ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(ref_time, ref_freq,
                                                                         est_time, est_freq,
                                                                         kind='zero', hop=0.01)
            for n, metric in enumerate([mir_eval.melody.overall_accuracy, mir_eval.melody.raw_pitch_accuracy,
                                        mir_eval.melody.raw_chroma_accuracy, mir_eval.melody.voicing_measures]):
                if metric == mir_eval.melody.voicing_measures:
                    vx_recall, vx_false_alm = mir_eval.melody.voicing_measures(ref_v, est_v)
                    scores[n] += vx_recall
                    scores[n + 1] += vx_false_alm
                else:
                    scores[n] += metric(ref_v, ref_c, est_v, est_c)
        # Compute mean
        scores /= float(N + 1)
        output_path = os.path.split(estimated_melody_file)[0].replace('estimated', 'mir_eval_scores')
        # Make sure output dir exists
        try:
            os.makedirs(output_path)
        except OSError:
            pass
        np.savetxt(estimated_melody_file.replace('estimated', 'mir_eval_scores'), scores)

# <codecell>

try:
    os.makedirs(os.path.join(BASE_DATA_PATH, 'mir_eval_scores'))
except OSError:
    pass
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
print ' & '.join(['{:10s}'.format(key) for key in METRIC_KEYS]),
print '\\\\'
print ' & '.join(['{:8.3f}\%'.format(score*100) for score in np.mean(np.abs(diff)/score_mean, axis=0)]),
print '\\\\'

# <codecell>

for n, key in enumerate(METRIC_KEYS):
    print ' '*len(os.path.split(score_glob[0])[1][25:]) + '\t' + ' '*10*n + key
for n, d in enumerate(diff):
    print os.path.split(score_glob[n])[1][25:] + '\t', '  '.join(["{:+.5f}".format(n) for n in d])

