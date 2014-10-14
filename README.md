Repository for all experiments/paper for mir_eval ISMIR 2014 submission.

Anything related to the paper goes in the paper subdirectory.

Experiments (comparing mir_eval against pre-existing eval implementations) are in the code folder.
To run each experiment, just run (e.g.)
python beat.py
from the experiments subdirectory.
You'll need mir_eval, numpy/scipy, and joblib to run the experiments.
Some of the experiments require downloading data from the NEMA servers, so require an internet connection (and some time).
Each experiment produces the average relative change figures reported in the paper in Table 2.
