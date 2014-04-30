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
import scipy.interpolate

# <codecell>

BASE_DATA_PATH = '../../data/melody'
METRIC_KEYS = ['Overall Accuracy', 'Raw Pitch Accuracy', 'Raw Chroma Accuracy', 'Voicing Recall Rate', 'Voicing False-Alarm Rate']
ALG_NAMES = ['SG2']

# <codecell>

# Download MIREX's reported files, which are subsampled to 50ms
subsampled_dir = os.path.join(BASE_DATA_PATH, 'raw_50ms_js')
try:
    os.makedirs(subsampled_dir)
except OSError:
    pass
JS_URL = 'http://nema.lis.illinois.edu/nema_out/mirex2011/results/ame/adc04/sg2melodytranscriptmeladc040000{:02d}.js'
for n in xrange(12):
    file_url = JS_URL.format(n)
    urllib.urlretrieve(file_url, os.path.join(subsampled_dir, os.path.split(file_url)[1]))
JS_URL = 'http://nema.lis.illinois.edu/nema_out/mirex2011/results/ame/adc04/sg2melodytranscriptmeladc04nv00000{}.js'
for n in xrange(8):
    file_url = JS_URL.format(n)
    urllib.urlretrieve(file_url, os.path.join(subsampled_dir, os.path.split(file_url)[1]))

# <codecell>

# Maps between ADC2004 names and MIREX names
FILENAME_MAPPING = {'opera_male5REF.txt' : 'sg2melodytranscriptmeladc04000000.js',
                    'daisy3REF.txt' : 'sg2melodytranscriptmeladc04000001.js',
                    'daisy1REF.txt' : 'sg2melodytranscriptmeladc04000002.js',
                    'daisy2REF.txt' : 'sg2melodytranscriptmeladc04000003.js',
                    'opera_male3REF.txt' : 'sg2melodytranscriptmeladc04000004.js',
                    'daisy4REF.txt' : 'sg2melodytranscriptmeladc04000005.js',
                    'pop1REF.txt' : 'sg2melodytranscriptmeladc04000006.js',
                    'opera_fem2REF.txt' : 'sg2melodytranscriptmeladc04000007.js',
                    'pop2REF.txt' : 'sg2melodytranscriptmeladc04000008.js',
                    'pop3REF.txt' : 'sg2melodytranscriptmeladc04000009.js',
                    'opera_fem4REF.txt' : 'sg2melodytranscriptmeladc04000010.js',
                    'pop4REF.txt' : 'sg2melodytranscriptmeladc04000011.js',
                    'midi4REF.txt' : 'sg2melodytranscriptmeladc04nv000000.js',
                    'midi3REF.txt' : 'sg2melodytranscriptmeladc04nv000001.js',
                    'midi2REF.txt' : 'sg2melodytranscriptmeladc04nv000002.js',
                    'midi1REF.txt' : 'sg2melodytranscriptmeladc04nv000003.js',
                    'jazz1REF.txt' : 'sg2melodytranscriptmeladc04nv000004.js',
                    'jazz2REF.txt' : 'sg2melodytranscriptmeladc04nv000005.js',
                    'jazz3REF.txt' : 'sg2melodytranscriptmeladc04nv000006.js',
                    'jazz4REF.txt' : 'sg2melodytranscriptmeladc04nv000007.js'}

# <codecell>

for ref_file in glob.glob(os.path.join(BASE_DATA_PATH, 'reference', '*')):
    mapped_name = FILENAME_MAPPING[os.path.split(ref_file)[1]].replace('.js', '.txt')
    shutil.copy(ref_file, ref_file.replace(os.path.split(ref_file)[1], mapped_name))
    est_file = ref_file.replace('reference', 'estimated').replace('REF.txt', '_mel.txt')
    shutil.copy(est_file, est_file.replace(os.path.split(est_file)[1], mapped_name))

# <codecell>

import re
import json
def convert_json_file_to_lab_files(base_name, js_dir, lab_dir):
    # Read js-file and store into single-line string
    js_path = os.path.join(js_dir, base_name+'.js')
    with open(js_path) as js_file:
        js_content = js_file.read()
        js_content = js_content.replace('\n','')
        
        # Use regexp to isolate json
        data_exp = re.compile('var .*_data = ([^;]*);')
        names_exp = re.compile('var .*_seriesNames = ([^;]*);')
        data_match = data_exp.search(js_content)
        names_match = names_exp.search(js_content)
        names = names_match.group(1)		
        data = data_match.group(1)
        data = data.replace('{x: ', '[').replace('y: ', '').replace('}', ']')
        # Parse json
        n = json.loads(names)
        d = json.loads(data)
        
        # Write lab file for each series
        for j in range(len(n)):
            name_dir = os.path.join(lab_dir, n[j])
            if not os.path.exists(name_dir):
                os.makedirs(name_dir)
            lab_path = os.path.join(name_dir, base_name+'.lab')
            with open(lab_path, 'w') as lab_file:
                for k in range(len(d[j])):
                    lab_file.write('{0}\n'.format('\t'.join([str(m) for m in d[j][k]])))

# <codecell>

mirex_reference_dir = os.path.join(BASE_DATA_PATH, 'reference_mirex_50ms')

try:
    os.makedirs(mirex_reference_dir)
except OSError:
    pass

for filename in glob.glob(os.path.join(subsampled_dir, '*.js')):
    base_name = os.path.splitext(os.path.split(filename)[1])[0]
    convert_json_file_to_lab_files(base_name, subsampled_dir, mirex_reference_dir)

for filename in glob.glob(os.path.join(reference_dir, '*', '*.lab')):
    shutil.move(filename, filename.replace('.lab', '.txt'))
    
mirex_estimated_dir = os.path.join(BASE_DATA_PATH, 'estimated_mirex_50ms')

try:
    os.makedirs(mirex_estimated_dir)
except OSError:
    pass

shutil.move(os.path.join(mirex_reference_dir, 'Prediction'), os.path.join(mirex_estimated_dir, 'SG2'))

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

for ds in ['reference', 'estimated']:
    for mirex_file in glob.glob(os.path.join(BASE_DATA_PATH, '{}_mirex_50ms'.format(ds), '*', '*.txt')):
        mirex_times, mirex_values = mir_eval.io.load_time_series(mirex_file)
        mir_eval_times, mir_eval_values = mir_eval.io.load_time_series(mirex_file.replace('{}_mirex_50ms'.format(ds),
                                                                                          '{}'.format(ds)))
        mir_eval_values_resampled = resample_zero(mir_eval_times, mir_eval_values, mirex_times)
        error = np.abs(mirex_values - mir_eval_values_resampled)
        error_inds = np.flatnonzero(error > .01)
        if error_inds.size > 0:
            print "File:", ds, os.path.split(mirex_file)[1]
            error = np.abs(mirex_values - mir_eval_values_resampled)
            error_inds = np.flatnonzero(error > .01)
            print "Time series length:", error.size
            print "Resampled indices with error > .01:"
            print error_inds
            print "Error at these indices:"
            print error[error_inds]
            print "Original time and resampled time are the same?"
            print (np.min(np.abs(np.subtract.outer(mirex_times[error_inds], mir_eval_times)), axis=1) == 0)*1
            print

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
            ref_v, est_v, ref_c, est_c = mir_eval.melody.to_cent_voicing(ref_time, ref_freq,
                                                                         est_time, est_freq,
                                                                         kind='zero', hop=0.01)
            for n, metric_name in enumerate(METRIC_KEYS):
                if metric_name == 'Voicing Recall Rate':
                    vx_recall, vx_false_alm = mir_eval.melody.voicing_measures(ref_v, est_v)
                    scores[n] += vx_recall
                    scores[n + 1] += vx_false_alm
                elif metric_name == 'Voicing False-Alarm Rate':
                    continue
                else:
                    scores[n] += mir_eval.melody.METRICS[metric_name](ref_v, est_v, ref_c, est_c)
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

print "Relative"
print ' ',
for n, key in enumerate(METRIC_KEYS):
    print '{:13s}'.format(key[:12]),
print
print np.sum(np.abs(diff), axis=0)/np.sum(mirex_scores, axis=0)
print
print "Absolute"
print ' ',
for n, key in enumerate(METRIC_KEYS):
    print '{:13s}'.format(key[:12]),
print
print np.mean(np.abs(diff), axis=0)/100.

# <codecell>

for n, key in enumerate(METRIC_KEYS):
    print ' '*len(os.path.split(score_glob[0])[1][25:]) + '\t' + ' '*10*n + key
for n, d in enumerate(diff):
    print os.path.split(score_glob[n])[1][25:] + '\t', '  '.join(["{:+.5f}".format(n) for n in d])

