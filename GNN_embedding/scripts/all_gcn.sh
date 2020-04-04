#!/bin/bash
#python3 train.py --network-type GCNConv --dataset Decagon --use-features
#python3 train.py --network-type GCNConv --dataset Decagon_GNBR --use-features
python3 train.py --network-type GCNConv --dataset Decagon_GNBR
python3 train.py --network-type GCNConv --dataset Decagon
python3 train.py --network-type GCNConv --dataset GNBR --use-features
python3 train.py --network-type GCNConv --dataset GNBR
