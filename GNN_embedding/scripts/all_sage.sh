#!/bin/bash
python3 train.py --network-type SAGEConv --dataset Decagon --use-features
python3 train.py --network-type SAGEConv --dataset Decagon_GNBR --use-features
python3 train.py --network-type SAGEConv --dataset Decagon_GNBR
python3 train.py --network-type SAGEConv --dataset Decagon
python3 train.py --network-type SAGEConv --dataset GNBR --use-features
python3 train.py --network-type SAGEConv --dataset GNBR
