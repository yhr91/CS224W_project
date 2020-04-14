#!/bin/bash
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 500 --sample-diseases True --hidden-dim 36 --shuffle True --expt_name HO_MT_epochs_500
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 1000 --sample-diseases True --hidden-dim 36 --shuffle True --expt_name HO_MT_epochs_1000
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 1500 --sample-diseases True --hidden-dim 36 --shuffle True --expt_name HO_MT_epochs_1500
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --hidden-dim 36 --shuffle True --expt_name HO_MT_epochs_2000
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 3000 --sample-diseases True --hidden-dim 36 --shuffle True --expt_name HO_MT_epochs_3000
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 3000 --sample-diseases True --hidden-dim 36 --shuffle True --lr 0.0001 --expt_name HO_MT_epochs_3000_0.0001
