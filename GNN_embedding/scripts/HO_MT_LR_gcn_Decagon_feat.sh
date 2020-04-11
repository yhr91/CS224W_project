#!/bin/bash
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --lr 0.001 --expt_name HO_MT_LR_0.001
python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --lr 0.0001 --expt_name HO_MT_LR_0.0001
python3 train.py --network-type SAGE --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --lr 0.001 --expt_name HO_MT_LR_0.001
python3 train.py --network-type SAGE --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --lr 0.0001 --expt_name HO_MT_LR_0.0001
python3 train.py --network-type SAGE_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --lr 0.001 --expt_name HO_MT_LR_0.001
python3 train.py --network-type SAGE_GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --lr 0.0001 --expt_name HO_MT_LR_0.0001
python3 train.py --network-type GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --lr 0.001 --expt_name HO_MT_LR_0.001
python3 train.py --network-type GCN --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --lr 0.0001 --expt_name HO_MT_LR_0.0001
python3 train.py --network-type GEO_GAT --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --lr 0.001 --expt_name HO_MT_LR_0.001
python3 train.py --network-type GEO_GAT --dataset Decagon --use-features --MTL True --epochs 2000 --sample-diseases True --lr 0.0001 --expt_name HO_MT_LR_0.0001
