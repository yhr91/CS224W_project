#!/bin/bash
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --hidden-dim 12 --expt_name HO_MT_HL_12
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --hidden-dim 24 --expt_name HO_MT_HL_24
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --hidden-dim 36 --expt_name HO_MT_HL_36
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --hidden-dim 48 --expt_name HO_MT_HL_48
python3 train.py --network-type SAGE_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --hidden-dim 12 --expt_name HO_MT_HL_12
python3 train.py --network-type SAGE_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --hidden-dim 24 --expt_name HO_MT_HL_24
python3 train.py --network-type SAGE_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --hidden-dim 36 --expt_name HO_MT_HL_36
python3 train.py --network-type SAGE_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --hidden-dim 48 --expt_name HO_MT_HL_48
