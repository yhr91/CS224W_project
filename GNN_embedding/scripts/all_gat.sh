#!/bin/bash
python3 train.py --network-type GATConv --dataset Decagon --use-features --num-heads 3
python3 train.py --network-type GATConv --dataset Decagon_GNBR --use-features --num-heads 3
python3 train.py --network-type GATConv --dataset Decagon_GNBR --num-heads 3
python3 train.py --network-type GATConv --dataset Decagon --num-heads 3
python3 train.py --network-type GATConv --dataset GNBR --use-features --num-heads 3
python3 train.py --network-type GATConv --dataset GNBR --num-heads 3
