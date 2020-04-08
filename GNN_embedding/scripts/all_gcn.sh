#!/bin/bash
START=$(date +%s.%N)
python3 train.py --network-type GCNConv --dataset Decagon --use-features
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)

START=$(date +%s.%N)
python3 train.py --network-type GCNConv --dataset Decagon_GNBR --use-features
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)

START=$(date +%s.%N)
python3 train.py --network-type GCNConv --dataset Decagon_GNBR
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)

START=$(date +%s.%N)
python3 train.py --network-type GCNConv --dataset Decagon
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)

START=$(date +%s.%N)
python3 train.py --network-type GCNConv --dataset GNBR --use-features
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)

START=$(date +%s.%N)
python3 train.py --network-type GCNConv --dataset GNBR
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)