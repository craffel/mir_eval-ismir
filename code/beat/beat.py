# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Compare mir_eval against MIREX 2013 beat results
'''

# <codecell>

import glob
import os
import csv
import mir_eval
import sys
sys.path.append('../')
import convert_json_instants_to_lab
import shutil
import joblib
import urllib
import numpy as np

# <codecell>

BASE_DATA_PATH = '../../data/beat/'
METRIC_KEYS = ['F-Measure', 'Cemgil', 'Goto', 'McKinney P-score', 'CMLc', 'CMLt', 'AMLc', 'AMLt', 'D (bits)']
ALG_NAMES = ['FW4', 'ZDBG1', 'FW2', 'FW1', 'KFRO1', 'ZDG1', 'ZDG2', 'CDF2', 'CDF1', 'GP3',
             'DP1', 'SB6', 'GP1', 'GKC3', 'SB5', 'GP2', 'ES3', 'ES1', 'EWFS1', 'FK1']

# <codecell>

datasets = ['dav', 'maz', 'mck']
js_prefixes = ['smc', 'beatmaz000', 'beatmck000']
# SMC dataset is missing a lot of entries.
file_ranges = [np.delete(np.arange(1, 290), [19,24,28,30,38,39,44,48,49,52,61,69,
                                             76,77,80,82,89,90,93,96,101,106,107,
                                             109,111,114,121,122,124,127,128,130,
                                             131,133,135,137,140,143,144,154,155,
                                             159,161,162,163,164,176,179,182,184,
                                             185,188,190,195,199,200,209,217,227,
                                             229,230,232,233,237,239,244,245,246,
                                             249,266,267,269]), range(322), range(140)]
JS_URL = "http://nema.lis.illinois.edu/nema_out/mirex2013/results/abt/{}/transcription{}{:03d}.js"
raw_js_dir = os.path.join(BASE_DATA_PATH, 'raw_js_data')
if not os.path.exists(raw_js_dir):
    os.makedirs(raw_js_dir)
for dataset, js_prefix, file_range in zip(datasets, js_prefixes, file_ranges):
    for n in file_range:
        file_url = JS_URL.format(dataset, js_prefix, n)
        urllib.urlretrieve(file_url, os.path.join(raw_js_dir, os.path.split(file_url)[1]))

# <codecell>

RESULT_URL = "http://nema.lis.illinois.edu/nema_out/mirex2013/results/abt/{}/{}/per_track_results.csv"
csv_dir = os.path.join(BASE_DATA_PATH, 'mirex_scores_raw')
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
for dataset in datasets:
    for alg in ALG_NAMES:
        file_url = RESULT_URL.format(dataset, alg)
        urllib.urlretrieve(file_url, os.path.join(csv_dir, dataset + '-' + alg + '.csv'))

# <codecell>

# A typical row in a MIREX results .csv looks like:
#0,smc_001,14.2860,13.9210,0.0000,35.7140,0.0000,0.0000,0.0000,0.0000,1.0967
# Where the order of the metrics is
# F-Measure,Cemgil,Goto,McKinney P-score,CMLc,CMLt,AMLc,AMLt,D (bits)
# The first column is "fold", which is not useful for us.
# The second column is the file, but it's not the filename.
# So, remove the first column and make the second look like the filename
for filename in glob.glob(os.path.join(BASE_DATA_PATH, 'mirex_scores_raw', '*.csv')):
    output_list = []
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            # Skip the label row
            if row[0] != '*Fold':
                # Construct the filename/output path
                output_filename = 'transcription' + row[1].replace('_', '') + '.txt'
                output_path = filename.replace('mirex_scores_raw', 'mirex_scores')
                algo_name = os.path.split(output_path)[1].replace('.csv', '').split('-')[1]
                output_path = os.path.join(os.path.split(output_path)[0], algo_name)
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

def get_mir_eval_scores(reference_beats, estimated_beats):
    ''' Computes all mir_eval metrics and returns them in a list with the order
    F-Measure,Cemgil,Goto,McKinney P-score,CMLc,CMLt,AMLc,AMLt,D (bits)
    '''
    scores = []
    for metric_name in ['F-measure', 'Cemgil', 'Goto', 'P-score', 'Continuity', 'Information Gain']:
        
        metric_output = mir_eval.beat.METRICS[metric_name](np.sort(mir_eval.beat.trim_beats(reference_beats)),
                                                           np.sort(mir_eval.beat.trim_beats(estimated_beats)))
        # MIREX only reports Cemgil for no metric variations
        if metric_name == 'Cemgil':
            scores.append(metric_output[0])
        # MIREX reports unnormalized information gain, we normalize by log2(41)
        elif metric_name == 'Information Gain':
            scores.append(metric_output*np.log2(41.)/100.)
        else:
            # Some metrics return tuples
            try:
                scores += list(metric_output)
            # Some return floats
            except TypeError:
                scores.append(metric_output)
    # Mirex reports scores on a 0-100 scale
    return np.array(scores)*100.

# <codecell>

def process_one_algorithm(algorithm_directory, skip=False):
    ''' Computes mir_eval scores for all output files from one algorithm '''
    for estimated_beats_file in glob.glob(os.path.join(BASE_DATA_PATH, 'estimated',
                                                       algorithm_directory, '*.txt')):
        estimated_beats = np.loadtxt(estimated_beats_file)
        # Skip scores already computed
        if skip and os.path.exists(estimated_beats_file.replace('estimated', 'mir_eval_scores')):
            continue
        scores = np.zeros(len(METRIC_KEYS))
        # Metrics are computed as the mean across all reference annotations
        for N, reference_beats_file in enumerate(glob.glob(os.path.join(BASE_DATA_PATH, 'reference', '*',
                                                                        os.path.split(estimated_beats_file)[1]))):
            reference_beats = np.loadtxt(reference_beats_file)
            scores += get_mir_eval_scores(reference_beats, estimated_beats)
        # Compute mean
        scores /= float(N + 1)
        output_path = os.path.split(estimated_beats_file)[0].replace('estimated', 'mir_eval_scores')
        # Make sure output dir exists
        try:
            os.makedirs(output_path)
        except OSError:
            pass
        np.savetxt(estimated_beats_file.replace('estimated', 'mir_eval_scores'), scores)

# <codecell>

joblib.Parallel(n_jobs=7)(joblib.delayed(process_one_algorithm)(algo) for algo in ALG_NAMES)

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

np.set_printoptions(precision=3, threshold=10000, linewidth=150, suppress=True)

# <codecell>

score_mean = np.mean(np.dstack([np.round(mirex_scores, 3), np.round(mir_eval_scores, 3)]), axis=-1)
score_mean = score_mean + (score_mean == 0)

# <codecell>

diff = np.round(mirex_scores, 3) - np.round(mir_eval_scores, 3)
diff[np.less_equal(np.abs(diff), .0010001)] = 0
print ' & '.join(['{:10s}'.format(key.replace('McKinney', '')) for key in METRIC_KEYS]),
print '\\\\'
print ' & '.join(['{:8.3f}\%'.format(score*100) for score in np.mean(np.abs(diff)/score_mean, axis=0)]),
print '\\\\'

