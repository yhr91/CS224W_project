#!/bin/bash
#python3 train.py --network-type ADA_GCN --use-features --MTL --epochs 2000 --score loss_sum --hidden-dim 36 --shuffle --sample-diseases --expt_name graphcombo_compare_hetero
python3 train.py --network-type GEO_GCN --dataset Decagon_GNBR --use-features --MTL --epochs 2000 --score loss_sum --hidden-dim 36 --shuffle --sample-diseases --expt_name graphcombo_compare_naive
#python3 train.py --network-type GEO_GCN --dataset GNBR --use-features --MTL --epochs 2000 --score loss_sum --hidden-dim 36 --shuffle --sample-diseases --expt_name graphcombo_compare_GNBR
#python3 train.py --network-type GEO_GCN --dataset Decagon --use-features --MTL --epochs 2000 --score loss_sum --hidden-dim 36 --shuffle --sample-diseases --expt_name graphcombo_compare_Decagon
#python3 train.py --network-type NO_GNN --dataset Decagon --use-features --MTL --epochs 2000 --score loss_sum --hidden-dim 36 --shuffle --sample-diseases --expt_name graphcombo_compare_noedge
