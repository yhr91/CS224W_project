#!/bin/bash
python3 train.py --network-type HGCNConv --dataset Decagon --use-features
python3 train.py --network-type HGCNConv --dataset Decagon_GNBR --use-features
python3 train.py --network-type HGCNConv --dataset Decagon_GNBR
python3 train.py --network-type HGCNConv --dataset Decagon
python3 train.py --network-type HGCNConv --dataset GNBR --use-features
python3 train.py --network-type HGCNConv --dataset GNBR
