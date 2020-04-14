#!/bin/bash
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases False --hidden-dim 36 --shuffle --expt_name HO_MT_shuf_True
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases False --hidden-dim 36 --expt_name HO_MT_shuf_False
