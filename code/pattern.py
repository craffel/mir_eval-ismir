# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob
import os
import sys
import mir_eval
import numpy as np

# <codecell>

BASE_DATA_PATH = '../data/pattern/'
METRIC_KEYS = ['P_est', 'R_est', 'F1_est', 'P_occ(c=.75)', 'R_occ(c=.75)', 'F_1occ(c=.75)', 'P_3', 'R_3',
               'TLF_1', 'FFTP_est', 'FFP', 'P_occ(c=.5)', 'R_occ(c=.5)', 'F_1occ(c=.5)', 'P', 'R', 'F_1']

# <codecell>

# Convert flat .txt to dirs
current_algorithm = ''
current_file = ''
with open(os.path.join(BASE_DATA_PATH, 'matlab_scores_raw', 'resultsMATLAB.txt')) as f:
    for line in f:
        if 'Algorithm' in line:
            current_algorithm = line.strip()[-3:]
        elif 'monophonic' in line:
            current_phonic = 'monophonic'
            current_file = line.strip().replace('-monophonic', '')
        elif 'polyphonic' in line:
            current_phonic = 'polyphonic'
            current_file = line.strip().replace('-polyphonic', '')
        else:
            try:
                scores = np.array([float(n) for n in line.strip().split(',')])
            except ValueError:
                continue
            output_path = os.path.join(BASE_DATA_PATH, current_phonic, 'matlab_scores', current_algorithm)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            np.savetxt(os.path.join(output_path, current_file + '.txt'), scores)

# <codecell>

def get_mir_eval_scores(reference_pattern, estimated_pattern):
    ''' Computes all mir_eval pattern metrics and returns them in the correct order
    '''
    #METRIC_KEYS = ['P_est', 'R_est', 'F1_est', 'P_occ(c=.75)', 'R_occ(c=.75)', 'F_1occ(c=.75)', 'P_3', 'R_3',
    #               'TLF_1', 'FFTP_est', 'FFP', 'P_occ(c=.5)', 'R_occ(c=.5)', 'F_1occ(c=.5)', 'P', 'R', 'F_1'    scores = []
    '''
    ['standard_FPR',
     'establishment_FPR',
     'occurrence_FPR',
     'three_layer_FPR',
     'first_n_three_layer_P',
     'first_n_target_proportion_R']
    '''
    scores = np.zeros(17)
    F1_est, P_est, R_est = mir_eval.pattern.establishment_FPR(reference_pattern, estimated_pattern)
    scores[0] = P_est
    scores[1] = R_est
    scores[2] = F1_est
    F1_occ, P_occ, R_occ = mir_eval.pattern.occurrence_FPR(reference_pattern, estimated_pattern, .75)
    scores[3] = P_occ
    scores[4] = R_occ
    scores[5] = F1_occ
    TLF_1, P_3, R_3 = mir_eval.pattern.three_layer_FPR(reference_pattern, estimated_pattern)
    scores[6] = P_3
    scores[7] = R_3
    scores[8] = TLF_1
    scores[9] = mir_eval.pattern.first_n_target_proportion_R(reference_pattern, estimated_pattern)
    scores[10] = mir_eval.pattern.first_n_three_layer_P(reference_pattern, estimated_pattern)
    F1_occ, P_occ, R_occ = mir_eval.pattern.occurrence_FPR(reference_pattern, estimated_pattern, .5)
    scores[11] = P_occ
    scores[12] = R_occ
    scores[13] = F1_occ
    F_1, P, R = mir_eval.pattern.standard_FPR(reference_pattern, estimated_pattern)
    scores[14] = P
    scores[15] = R
    scores[16] = F_1
    
    return scores

# <codecell>

def process_one_algorithm(algorithm_directory, skip=False):
    ''' Computes mir_eval scores for all output files from one algorithm '''
    for estimated_pattern_file in glob.glob(os.path.join(algorithm_directory, '*.txt')):
        estimated_pattern = mir_eval.io.load_patterns(estimated_pattern_file)
        # Skip scores already computed
        if skip and os.path.exists(estimated_pattern_file.replace('estimated', 'mir_eval_scores')):
            continue
        scores = np.zeros(len(METRIC_KEYS))
        # Metrics are computed as the mean across all reference annotations
        ref_glob = os.path.join(os.path.split(algorithm_directory.replace('estimated', 'reference'))[0],
                                '*', os.path.split(estimated_pattern_file)[1])
        for N, reference_pattern_file in enumerate(glob.glob(ref_glob)):
            reference_pattern = mir_eval.io.load_patterns(reference_pattern_file)
            try:
                scores += get_mir_eval_scores(reference_pattern, estimated_pattern)
            except Exception as e:
                N = N - 1
                print "Skipping {} because {}".format(os.path.split(reference_pattern_file)[1], e)
        if N >= 0:
            # Compute mean
            scores /= float(N + 1)
            output_path = os.path.split(estimated_pattern_file)[0].replace('estimated', 'mir_eval_scores')
            # Make sure output dir exists
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            np.savetxt(estimated_pattern_file.replace('estimated', 'mir_eval_scores'), scores)

# <codecell>

mir_eval_score_path = os.path.join(BASE_DATA_PATH, 'monophonic', 'mir_eval_scores')
if not os.path.exists(mir_eval_score_path):
    os.makedirs(mir_eval_score_path)
for alg_name in ['nf1', 'nf3']:
    process_one_algorithm(os.path.join(BASE_DATA_PATH, 'monophonic', 'estimated', alg_name))
mir_eval_score_path = os.path.join(BASE_DATA_PATH, 'polyphonic', 'mir_eval_scores')
if not os.path.exists(mir_eval_score_path):
    os.makedirs(mir_eval_score_path)
for alg_name in ['nf2', 'nf4']:
    process_one_algorithm(os.path.join(BASE_DATA_PATH, 'polyphonic', 'estimated', alg_name))

# <codecell>

mir_eval_scores = []
mirex_scores = []
score_glob = glob.glob(os.path.join(BASE_DATA_PATH, '*', 'mir_eval_scores', '*', '*.txt'))
for mir_eval_score_file in score_glob:
    mir_eval_scores.append([np.loadtxt(mir_eval_score_file)])
    mirex_scores.append([np.loadtxt(mir_eval_score_file.replace('mir_eval_scores', 'matlab_scores'))])
mir_eval_scores = np.vstack(mir_eval_scores)
mirex_scores = np.vstack(mirex_scores)    

# <codecell>

np.set_printoptions(precision=10, threshold=10000, linewidth=150, suppress=True)

# <codecell>

score_mean = np.mean(np.dstack([np.round(mirex_scores, 4), np.round(mir_eval_scores, 4)]), axis=-1)
score_mean = score_mean + (score_mean == 0)

# <codecell>

diff = np.round(mirex_scores, 4) - np.round(mir_eval_scores, 4)
diff[np.less_equal(np.abs(diff), .00010001)] = 0
relative_change = np.mean(np.abs(diff)/score_mean, axis=0)
for metric_name, change in zip(METRIC_KEYS, relative_change):
    print metric_name, ':', change*100, '%'

