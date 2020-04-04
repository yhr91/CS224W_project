#!/bin/bash
python3 train.py --network-type SAGEConvMean --dataset Decagon --use-features
python3 train.py --network-type SAGEConvMean --dataset Decagon_GNBR --use-features
python3 train.py --network-type SAGEConvMean --dataset Decagon_GNBR
python3 train.py --network-type SAGEConvMean --dataset Decagon
python3 train.py --network-type SAGEConvMean --dataset GNBR --use-features
python3 train.py --network-type SAGEConvMean --dataset GNBR
