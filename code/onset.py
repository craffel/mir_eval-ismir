# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import urllib
import glob
import os
import convert_json_instants_to_lab
import shutil
import csv
import mir_eval
import numpy as np

# <codecell>

BASE_DATA_PATH = '../data/onset/'

METRIC_KEYS = ['Average F-Measure', 'Average precision', 'Average recall']
ALG_NAMES = ['CB1','CF4','CSF1','FMEGS1','FMESS1','MTB1','SB1','SB2','SB3','SB4','ZHZD1']

# <codecell>

if not os.path.exists(BASE_DATA_PATH):
    os.makedirs(BASE_DATA_PATH)

js_dir = os.path.join(BASE_DATA_PATH, 'raw_js_data')
reference_dir = os.path.join(BASE_DATA_PATH, 'reference')
csv_dir = os.path.join(BASE_DATA_PATH, 'mirex_scores_raw')
estimated_dir = os.path.join(BASE_DATA_PATH, 'estimated')

if not os.path.exists(js_dir):
    os.makedirs(js_dir)
if not os.path.exists(reference_dir):
    os.makedirs(reference_dir)
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
if not os.path.exists(os.path.join(BASE_DATA_PATH, 'mir_eval_scores')):
    os.makedirs(os.path.join(BASE_DATA_PATH, 'mir_eval_scores'))

# <codecell>

N_FILES = 85
JS_URL = "http://nema.lis.illinois.edu/nema_out/mirex2013/results/aod/transcriptiononsetmrx050000{:02d}.js"
for n in xrange(N_FILES):
    file_url = JS_URL.format(n)
    urllib.urlretrieve(file_url, os.path.join(js_dir, os.path.split(file_url)[1]))

# <codecell>

RESULT_URL = "http://nema.lis.illinois.edu/nema_out/mirex2013/results/aod/{}/per_track_results.csv"
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
                output_filename = 'transcription' + row[1].replace('_', '') + '.txt'
                output_path = filename.replace('mirex_scores_raw', 'mirex_scores')
                output_path = os.path.join(os.path.split(output_path)[0], os.path.split(output_path)[1].replace('.csv', ''))
                # Make sure output dir exists
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                np.savetxt(os.path.join(output_path, output_filename), [float(x) for x in row[2:]])

# <codecell>

for filename in glob.glob(os.path.join(js_dir, '*.js')):
    base_name = os.path.splitext(os.path.split(filename)[1])[0]
    convert_json_instants_to_lab.convert_json_file_to_lab_files(base_name, js_dir, reference_dir)

for filename in glob.glob(os.path.join(reference_dir, '*', '*.lab')):
    shutil.move(filename, filename.replace('.lab', '.txt'))
    
for alg_name in ALG_NAMES:
    shutil.move(os.path.join(reference_dir, alg_name), os.path.join(estimated_dir, alg_name))

# <codecell>

def clean_onsets(onsets):
    ''' Turns 0-dim onset lists (floats) into 1-dim arrays '''
    if onsets.ndim == 0:
        return np.array([onsets])
    else:
        return onsets

# <codecell>

def process_one_algorithm(algorithm_directory, skip=False):
    ''' Computes mir_eval scores for all output files from one algorithm '''
    for estimated_onsets_file in glob.glob(os.path.join(BASE_DATA_PATH, 'estimated',
                                                        algorithm_directory, '*.txt')):
        estimated_onsets = clean_onsets(np.loadtxt(estimated_onsets_file))
        # Skip scores already computed
        if skip and os.path.exists(estimated_onsets_file.replace('estimated', 'mir_eval_scores')):
            continue
        scores = np.zeros(len(METRIC_KEYS))
        # Metrics are computed as the mean across all reference annotations
        for N, reference_onsets_file in enumerate(glob.glob(os.path.join(reference_dir, '*',
                                                                         os.path.split(estimated_onsets_file)[1]))):
            reference_onsets = clean_onsets(np.loadtxt(reference_onsets_file))
            scores += mir_eval.onset.f_measure(reference_onsets, estimated_onsets)
        # Compute mean
        scores /= float(N + 1)
        output_path = os.path.split(estimated_onsets_file)[0].replace('estimated', 'mir_eval_scores')
        # Make sure output dir exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.savetxt(estimated_onsets_file.replace('estimated', 'mir_eval_scores'), scores)

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
print ' & '.join(['{:10s}'.format(key) for key in ['F-measure', 'Precision', 'Recall']]),
print '\\\\'
print ' & '.join(['{:8.3f}\%'.format(score*100) for score in np.mean(np.abs(diff)/score_mean, axis=0)]),
print '\\\\'

