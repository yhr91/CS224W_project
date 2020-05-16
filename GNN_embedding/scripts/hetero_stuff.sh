python train.py --network-type ADA_A --use-features --MTL --sample-diseases --expt_name ADA_A
python train.py --network-type ADA_B --use-features --MTL --sample-diseases --expt_name ADA_B
python train.py --network-type ADA_C --use-features --MTL --sample-diseases --expt_name ADA_C
python train.py --network-type ADA_D --use-features --MTL --sample-diseases --expt_name ADA_D
python train.py --network-type ADA_D --use-features --MTL --sample-diseases --expt_name ADA_D2 --hidden-dim 16
python train.py --network-type ADA_E --use-features --MTL --sample-diseases --expt_name ADA_E
python train.py --network-type ADA_E --use-features --MTL --sample-diseases --expt_name ADA_E2 --hidden-dim 16
python train.py --network-type SAGE_GCN --dataset Decagon_GNBR --use-features --MTL --sample-diseases --expt_name SAGE_GCN
python train.py --network-type GEO_GCN --dataset Decagon_GNBR --use-features --MTL --sample-diseases --expt_name GEO_GCN
python train.py --network-type GCN --dataset Decagon_GNBR --use-features --MTL --sample-diseases --expt_name GCN