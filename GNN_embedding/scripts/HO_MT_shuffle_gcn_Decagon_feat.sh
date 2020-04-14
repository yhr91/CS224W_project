#!/bin/bash
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL --epochs 2000 --sample-diseases --hidden-dim 36 --shuffle --expt_name HO_MT_shuffle_True
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL --epochs 2000 --sample-diseases --hidden-dim 36 --expt_name HO_MT_shuffle_False
