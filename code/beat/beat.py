# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Compare mir_eval against MIREX 2012 beat results
'''

# <codecell>

import glob
import os
import csv
import mir_eval
import shutil
import joblib

# <codecell>

BASE_DATA_PATH = '../../data/beat/'
METRIC_KEYS = ['F-Measure', 'Cemgil', 'Goto', 'McKinney P-score', 'CMLc', 'CMLt', 'AMLc', 'AMLt', 'D (bits)']

# <codecell>

# A typical row in a MIREX results .csv looks like:
#0,smc_001,14.2860,13.9210,0.0000,35.7140,0.0000,0.0000,0.0000,0.0000,1.0967
# Where the order of the metrics is
# F-Measure,Cemgil,Goto,McKinney P-score,CMLc,CMLt,AMLc,AMLt,D (bits)
# The first column is "fold", which is not useful for us.
# The second column is the file, but it's not the filename.
# So, remove the first column and make the second look like the filename
for filename in glob.glob(os.path.join(BASE_DATA_PATH, '*', 'raw_scores', '*.csv')):
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
    for estimated_beats_file in glob.glob(os.path.join(BASE_DATA_PATH, '*', 'estimated',
                                                       algorithm_directory, '*.txt')):
        estimated_beats = np.loadtxt(estimated_beats_file)
        # Skip scores already computed
        if skip and os.path.exists(estimated_beats_file.replace('estimated', 'mir_eval_scores')):
            continue
        scores = np.zeros(len(METRIC_KEYS))
        # Metrics are computed as the mean across all reference annotations
        for N, reference_beats_file in enumerate(glob.glob(os.path.join(BASE_DATA_PATH, '*', 'reference', '*',
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

joblib.Parallel(n_jobs=7)(joblib.delayed(process_one_algorithm)(os.path.split(algo)[1])
                          for algo in glob.glob(os.path.join(BASE_DATA_PATH, 'maz', 'estimated', '*')))

# <codecell>

mir_eval_scores = []
mirex_scores = []
score_glob = glob.glob(os.path.join(BASE_DATA_PATH, '*', 'mir_eval_scores', '*', '*.txt'))
for mir_eval_score_file in score_glob:
    mir_eval_scores.append([np.loadtxt(mir_eval_score_file)])
    mirex_scores.append([np.loadtxt(mir_eval_score_file.replace('mir_eval_scores', 'mirex_scores'))])
mir_eval_scores = np.vstack(mir_eval_scores)
mirex_scores = np.vstack(mirex_scores)

# <codecell>

diff = np.abs(mirex_scores - np.round(mir_eval_scores, 3))
diff[np.less_equal(diff, .0010001)] = 0
print np.sum(diff, axis=0)/np.sum(mirex_scores, axis=0)
print np.sum(diff, axis=0)/mirex_scores.shape[0]

