# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import urllib
import glob
import os
import sys
sys.path.append('../')
import convert_json_instants_to_lab
import shutil
import csv
import mir_eval

# <codecell>

BASE_DATA_PATH = '../../data/onset/'
METRIC_KEYS = ['Average F-Measure', 'Average precision', 'Average recall']
ALG_NAMES = ['CB1','CF4','CSF1','FMEGS1','FMESS1','MTB1','SB1','SB2','SB3','SB4','ZHZD1']

# <codecell>

N_FILES = 85
JS_URL = "http://nema.lis.illinois.edu/nema_out/mirex2013/results/aod/transcriptiononsetmrx050000{:02d}.js"
for n in xrange(N_FILES):
    file_url = JS_URL.format(n)
    urllib.urlretrieve(file_url, os.path.join(BASE_DATA_PATH, 'raw_js_data', os.path.split(file_url)[1]))

# <codecell>

RESULT_URL = "http://nema.lis.illinois.edu/nema_out/mirex2013/results/aod/{}/per_track_results.csv"
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
                output_filename = 'transcription' + row[1].replace('_', '') + '.txt'
                output_path = filename.replace('mirex_scores_raw', 'mirex_scores')
                output_path = os.path.join(os.path.split(output_path)[0], os.path.split(output_path)[1].replace('.csv', ''))
                # Make sure output dir exists
                try:
                    os.makedirs(output_path)
                except OSError:
                    pass
                np.savetxt(os.path.join(output_path, output_filename), [float(x) for x in row[2:]])

# <codecell>

js_dir = os.path.join(BASE_DATA_PATH, 'raw_js_data')
reference_dir = os.path.join(BASE_DATA_PATH, 'reference')

try:
    os.makedirs(js_dir)
except OSError:
    pass
try:
    os.makedirs(reference_dir)
except OSError:
    pass

for filename in glob.glob(os.path.join(js_dir, '*.js')):
    base_name = os.path.splitext(os.path.split(filename)[1])[0]
    convert_json_instants_to_lab.convert_json_file_to_lab_files(base_name, js_dir, reference_dir)

for filename in glob.glob(os.path.join(reference_dir, '*', '*.lab')):
    shutil.move(filename, filename.replace('.lab', '.txt'))
    
estimated_dir = os.path.join(BASE_DATA_PATH, 'estimated')
        
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

def f_measure_greedy(reference_onsets, estimated_onsets, window=.05):
    # If either list is empty, return 0s
    if reference_onsets.size == 0 or estimated_onsets.size == 0:
        return 0., 0., 0.
    correct = 0.0
    count = 0
    for onset in reference_onsets:
        for n in xrange(count, estimated_onsets.shape[0]):
            if np.abs(estimated_onsets[n] - onset) < window:
                correct += 1
                count = n + 1
                break
            elif estimated_onsets[n] > (onset + window):
                count = n
                break
    precision = correct/estimated_onsets.shape[0]
    recall = correct/reference_onsets.shape[0]
    # Compute F-measure and return all statistics
    return mir_eval.util.f_measure(precision, recall), precision, recall

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
        for N, reference_onsets_file in enumerate(glob.glob(os.path.join(BASE_DATA_PATH, 'reference', '*',
                                                                         os.path.split(estimated_onsets_file)[1]))):
            reference_onsets = clean_onsets(np.loadtxt(reference_onsets_file))
            scores += mir_eval.onset.f_measure(reference_onsets, estimated_onsets)
        # Compute mean
        scores /= float(N + 1)
        output_path = os.path.split(estimated_onsets_file)[0].replace('estimated', 'mir_eval_scores')
        # Make sure output dir exists
        try:
            os.makedirs(output_path)
        except OSError:
            pass
        np.savetxt(estimated_onsets_file.replace('estimated', 'mir_eval_scores'), scores)

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

diff = np.abs(mirex_scores - np.round(mir_eval_scores, 4))
diff[np.less_equal(diff, .00010001)] = 0
print np.sum(diff, axis=0)/np.sum(mirex_scores, axis=0)
print np.mean(diff, axis=0)

