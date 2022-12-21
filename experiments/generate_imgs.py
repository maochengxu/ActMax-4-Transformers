import os, sys
parentPath = os.path.abspath("./")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

import torch
from transformers.tf_exp import ExpScores, ExpSyn
from core.insilico_exps import ExperimentEvolution

# 0 for tench and 574 for golf ball
model_unit = ("vit_b_16", ".heads.Linearhead", 0)

exp_syn = ExpSyn(model_unit, 'vit_b_16_exp', savedir='../results/evolutions/vit_b_16')

exp_syn.get_syn_images(50, 'tench')
