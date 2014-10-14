# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Compare mir_eval against MIREX 2013 segmentation results
'''

# <codecell>

import glob
import os
import csv
import mir_eval
import numpy as np
import convert_json_labels_to_lab
import shutil
import joblib
import urllib

# <codecell>

BASE_DATA_PATH = '../data/segment/'
METRIC_KEYS = ['Normalised conditional entropy based over-segmentation score',
               'Normalised conditional entropy based under-segmentation score',
               'Frame pair clustering F-measure',
               'Frame pair clustering precision rate',
               'Frame pair clustering recall rate',
               'Random clustering index',
               'Segment boundary recovery evaluation measure @ 0.5sec',
               'Segment boundary recovery precision rate @ 0.5sec',
               'Segment boundary recovery recall rate @ 0.5sec',
               'Segment boundary recovery evaluation measure @ 3sec',
               'Segment boundary recovery precision rate @ 3sec',
               'Segment boundary recovery recall rate @ 3sec',
               'Median distance from an annotated segment boundary to the closest found boundary',
               'Median distance from a found segment boundary to the closest annotated one']
ALG_NAMES = ['RBH3', 'RBH2', 'RBH1', 'MP1', 'RBH4', 'MP2', 'CF5', 'CF6']

# <codecell>

if not os.path.exists(BASE_DATA_PATH):
    os.makedirs(BASE_DATA_PATH)

js_dir = os.path.join(BASE_DATA_PATH, 'raw_js_data')
reference_dir = os.path.join(BASE_DATA_PATH, 'reference')
csv_dir = os.path.join(BASE_DATA_PATH, 'mirex_scores_raw')
raw_js_dir = os.path.join(BASE_DATA_PATH, 'raw_js_data')

if not os.path.exists(js_dir):
    os.makedirs(js_dir)
if not os.path.exists(reference_dir):
    os.makedirs(reference_dir)
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
if not os.path.exists(raw_js_dir):
    os.makedirs(raw_js_dir)

# <codecell>

datasets = ['mrx09', 'mrx10_1', 'mrx10_2', 'sal']
js_prefixes = ['structmrx09', 'structmrx10', 'structmrx10', 'salami']
# Salami is missing a bunch of entries
file_ranges = [range(297),
               range(100),
               range(100),
               np.delete(np.arange(1643),
                         [1, 2, 3, 4, 6, 12, 16, 17, 18, 19, 26, 27, 30, 31, 32, 38, 39, 40, 41,
                          43, 45, 49, 50, 51, 52, 57, 58, 59, 60, 61, 62, 65, 71, 72, 73, 74, 75,
                          76, 77, 78, 82, 84, 85, 87, 88, 100, 101, 102, 103, 104, 107, 108, 109,
                          110, 111, 112, 114, 115, 116, 117, 118, 124, 130, 133, 134, 136, 137,
                          138, 147, 149, 152, 154, 155, 156, 157, 159, 160, 164, 165, 167, 172,
                          174, 175, 177, 178, 179, 181, 184, 186, 189, 194, 195, 201, 206, 210,
                          211, 216, 228, 229, 230, 233, 236, 243, 250, 251, 257, 259, 260, 263,
                          266, 268, 271, 276, 278, 279, 282, 283, 284, 286, 287, 293, 295, 298,
                          303, 304, 305, 312, 314, 316, 317, 321, 324, 328, 330, 335, 336, 337,
                          338, 344, 345, 348, 349, 351, 352, 355, 356, 357, 359, 360, 361, 362,
                          364, 366, 376, 377, 382, 383, 384, 388, 394, 400, 405, 407, 408, 413,
                          416, 417, 418, 419, 421, 422, 428, 434, 437, 443, 448, 454, 459, 462,
                          464, 467, 469, 473, 474, 475, 476, 479, 481, 482, 483, 485, 486, 488,
                          489, 494, 495, 498, 501, 502, 503, 506, 508, 512, 519, 520, 522, 525,
                          530, 532, 538, 539, 541, 542, 543, 545, 546, 553, 558, 559, 561, 562,
                          565, 569, 572, 574, 576, 578, 579, 580, 581, 583, 584, 585, 586, 588,
                          589, 590, 594, 599, 605, 612, 617, 623, 634, 638, 641, 645, 646, 647,
                          648, 650, 655, 658, 661, 664, 665, 668, 671, 672, 674, 676, 677, 680,
                          685, 690, 692, 693, 697, 699, 702, 703, 705, 706, 708, 709, 710, 711,
                          712, 713, 716, 717, 720, 724, 725, 727, 728, 729, 730, 736, 741, 742,
                          743, 745, 748, 750, 756, 757, 763, 768, 774, 777, 778, 783, 785, 791,
                          793, 800, 802, 803, 805, 806, 809, 811, 812, 814, 815, 817, 823, 824,
                          826, 828, 829, 830, 833, 834, 835, 842, 848, 849, 850, 851, 856, 857,
                          858, 859, 861, 864, 865, 867, 868, 869, 871, 874, 876, 878, 880, 883,
                          885, 887, 892, 893, 896, 902, 903, 909, 911, 913, 916, 919, 926, 927,
                          930, 931, 933, 934, 935, 939, 940, 943, 945, 946, 950, 953, 956, 964,
                          966, 971, 972, 973, 977, 978, 979, 980, 981, 983, 984, 987, 989, 990,
                          991, 995, 1002, 1007, 1009, 1010, 1013, 1016, 1024, 1029, 1035, 1040,
                          1041, 1045, 1047, 1048, 1051, 1054, 1055, 1056, 1059, 1060, 1062, 1063,
                          1066, 1070, 1071, 1073, 1074, 1075, 1084, 1086, 1089, 1090, 1096, 1099,
                          1100, 1101, 1102, 1103, 1104, 1110, 1111, 1112, 1113, 1115, 1116, 1119,
                          1120, 1121, 1123, 1124, 1125, 1128, 1130, 1132, 1133, 1134, 1142, 1143,
                          1147, 1154, 1155, 1156, 1159, 1161, 1162, 1165, 1168, 1169, 1176, 1178,
                          1183, 1185, 1186, 1188, 1191, 1206, 1209, 1210, 1217, 1218, 1221, 1222,
                          1223, 1225, 1228, 1230, 1233, 1237, 1239, 1240, 1242, 1245, 1248, 1251,
                          1252, 1256, 1258, 1261, 1263, 1265, 1266, 1271, 1272, 1274, 1275, 1277,
                          1280, 1281, 1283, 1284, 1285, 1286, 1287, 1290, 1293, 1295, 1296, 1299,
                          1300, 1301, 1309, 1310, 1311, 1312, 1315, 1317, 1318, 1320, 1321, 1325,
                          1329, 1331, 1334, 1335, 1337, 1339, 1340, 1341, 1345, 1346, 1348, 1349,
                          1351, 1353, 1355, 1358, 1359, 1362, 1365, 1366, 1370, 1374, 1376, 1377,
                          1378, 1382, 1385, 1392, 1398, 1400, 1404, 1405, 1413, 1415, 1417, 1420,
                          1425, 1426, 1427, 1429, 1431, 1436, 1437, 1439, 1441, 1444, 1448, 1450,
                          1451, 1454, 1455, 1461, 1465, 1466, 1473, 1475, 1477, 1479, 1483, 1486,
                          1489, 1491, 1492, 1495, 1500, 1503, 1504, 1505, 1506, 1509, 1510, 1514,
                          1515, 1519, 1523, 1525, 1529, 1531, 1532, 1536, 1542, 1547, 1548, 1549,
                          1550, 1551, 1553, 1554, 1555, 1556, 1558, 1560, 1562, 1563, 1564, 1565,
                          1566, 1567, 1569, 1576, 1580, 1583, 1586, 1589, 1591, 1593, 1595, 1596,
                          1601, 1602, 1606, 1608, 1609, 1612, 1616, 1621, 1629, 1631, 1632, 1633,
                          1637, 1638])]
JS_URL = "http://nema.lis.illinois.edu/nema_out/mirex2013/results/struct/{}/segments{}{:06d}.js"
for dataset, js_prefix, file_range in zip(datasets, js_prefixes, file_ranges):
    for n in file_range:
        file_url = JS_URL.format(dataset, js_prefix, n)
        urllib.urlretrieve(file_url, os.path.join(raw_js_dir, os.path.split(file_url)[1]))

# <codecell>

RESULT_URL = "http://nema.lis.illinois.edu/nema_out/mirex2013/results/struct/{}/{}/per_track_results.csv"
for dataset in datasets:
    for alg in ALG_NAMES:
        file_url = RESULT_URL.format(dataset, alg)
        urllib.urlretrieve(file_url, os.path.join(csv_dir, dataset + '-' + alg + '.csv'))

# <codecell>

for filename in glob.glob(os.path.join(BASE_DATA_PATH, 'mirex_scores_raw', '*.csv')):
    output_list = []
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            # Skip the label row
            if row[0] != '*Fold':
                # Construct the filename/output path
                output_filename = 'segments' + row[1].replace('_', '') + '.txt'
                output_path = filename.replace('mirex_scores_raw', 'mirex_scores')
                algo_name = os.path.split(output_path)[1].replace('.csv', '').split('-')[1]
                output_path = os.path.join(os.path.split(output_path)[0], algo_name)
                # Make sure output dir exists
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                np.savetxt(os.path.join(output_path, output_filename), [float(x) for x in row[2:]])

# <codecell>

for filename in glob.glob(os.path.join(js_dir, '*.js')):
    base_name = os.path.splitext(os.path.split(filename)[1])[0]
    convert_json_labels_to_lab.convert_json_file_to_lab_files(base_name, js_dir, reference_dir)

for filename in glob.glob(os.path.join(reference_dir, '*', '*.lab')):
    shutil.move(filename, filename.replace('.lab', '.txt'))

estimated_dir = os.path.join(BASE_DATA_PATH, 'estimated')

for alg_name in ALG_NAMES:
    shutil.move(os.path.join(reference_dir, alg_name), os.path.join(estimated_dir, alg_name))

# <codecell>

# FK2 is not in salami, so we didn't get scores for it.
# We need to delete it in order for it not ot be left as reference.
shutil.rmtree(os.path.join(reference_dir, 'FK2'))

# <codecell>

def get_mir_eval_scores(ref_intervals, ref_labels, est_intervals, est_labels):
    ''' Computes all mir_eval segment metrics and returns them in the order of METRIC_KEYS
    '''
    scores = []
    ref_intervals_adj, ref_labels_adj = mir_eval.util.adjust_intervals(ref_intervals,
                                                                       ref_labels,
                                                                       t_min=0)
    est_intervals_adj, est_labels_adj = mir_eval.util.adjust_intervals(est_intervals,
                                                                       est_labels,
                                                                       t_min=0,
                                                                       t_max=ref_intervals.max())

    S_over, S_under, S_F = mir_eval.segment.nce(ref_intervals_adj, ref_labels_adj,
                                                  est_intervals_adj, est_labels_adj)
    scores.append(S_over)
    scores.append(S_under)

    precision, recall, f = mir_eval.segment.pairwise(ref_intervals_adj,
                                                       ref_labels_adj,
                                                       est_intervals_adj,
                                                       est_labels_adj)
    scores.append(f)
    scores.append(precision)
    scores.append(recall)

    rand_score = mir_eval.segment.rand_index(ref_intervals_adj, ref_labels_adj,
                                               est_intervals_adj, est_labels_adj)
    scores.append(rand_score)

    P05, R05, F05 = mir_eval.segment.detection(ref_intervals, est_intervals, window=0.5)
    scores.append(F05)
    scores.append(P05)
    scores.append(R05)
    P3, R3, F3 = mir_eval.segment.detection(ref_intervals, est_intervals, window=3)
    scores.append(F3)
    scores.append(P3)
    scores.append(R3)

    r_to_e, e_to_r = mir_eval.segment.deviation(ref_intervals, est_intervals)
    scores.append(r_to_e)
    scores.append(e_to_r)

    return np.array(scores)

# <codecell>

def process_one_algorithm(algorithm_directory, skip=False):
    ''' Computes mir_eval scores for all output files from one algorithm '''
    for estimated_segments_file in glob.glob(os.path.join(BASE_DATA_PATH, 'estimated',
                                                       algorithm_directory, '*.txt')):
        # Skip scores already computed
        if skip and os.path.exists(estimated_segments_file.replace('estimated', 'mir_eval_scores')):
            continue
        est_intervals, est_labels = mir_eval.io.load_labeled_intervals(estimated_segments_file)
        scores = np.zeros(len(METRIC_KEYS))
        # Metrics are computed as the mean across all reference annotations
        for N, reference_segments_file in enumerate(glob.glob(os.path.join(BASE_DATA_PATH, 'reference', '*',
                                                                        os.path.split(estimated_segments_file)[1]))):
            ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(reference_segments_file)
            try:
                scores += get_mir_eval_scores(ref_intervals, ref_labels, est_intervals, est_labels)
            except Exception as e:
                N = N - 1
                print "Skipping {} because {}".format(os.path.split(reference_segments_file)[1], e)
        if N >= 0:
            # Compute mean
            scores /= float(N + 1)
            output_path = os.path.split(estimated_segments_file)[0].replace('estimated', 'mir_eval_scores')
            # Make sure output dir exists
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            np.savetxt(estimated_segments_file.replace('estimated', 'mir_eval_scores'), scores)

# <codecell>

joblib.Parallel(n_jobs=8)(joblib.delayed(process_one_algorithm)(algo) for algo in ALG_NAMES)

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

np.set_printoptions(precision=5, threshold=10000, linewidth=150, suppress=True)

# <codecell>

score_mean = np.mean(np.dstack([np.round(mirex_scores, 4), np.round(mir_eval_scores, 4)]), axis=-1)
score_mean = score_mean + (score_mean == 0)

# <codecell>

diff = np.round(mirex_scores, 3) - np.round(mir_eval_scores, 3)
diff[np.less_equal(np.abs(diff), .0010001)] = 0
print ' & '.join(['{:10s}'.format(key) for key in ['NCE-Over', 'NCE-under', 'Pairwise F', 'Pairwise P',
                                                   'Pairwise R', 'Rand', 'F@.5', 'P@.5', 'R@.5',
                                                   'F@3', 'P@3', 'R@3', 'Ref-est dev.', 'Est-ref dev.']]),
print '\\\\'
print ' & '.join(['{:8.3f}\%'.format(score*100) for score in np.mean(np.abs(diff)/score_mean, axis=0)]),
print '\\\\'

