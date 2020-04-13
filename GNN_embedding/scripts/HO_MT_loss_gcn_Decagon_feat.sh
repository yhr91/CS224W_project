#!/bin/bash
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases False --hidden-dim 36 --score loss_sum --expt_name HO_MT_LO_LossSum
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases False --hidden-dim 36 --score loss_max --expt_name HO_MT_LO_LossMax
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases False --hidden-dim 36 --score f1_sum --expt_name HO_MT_LO_F1Sum
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases False --hidden-dim 36 --score f1_max --expt_name HO_MT_LO_F1Max
