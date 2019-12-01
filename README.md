## Graph embeddings of enriched protein-protein interaction (PPI) networks for identification of disease nodes

Goals:
Integration of traditionally distinct data modalities (including literature based knowledge graphs and multi-omics datasets) to enrich the information content of a PPI graph
Developing new methods for embedding these enriched graphs within a shared space that allows for better detection of disease modules in the graph



### Disease node classification leaderboard


| Model | Graph | Features | Other Comments | Accuracy* |
| --- | --- | --- | --- | --- |
| GCN | PP-Decagon | None | --- | --- |
| GCN | PP-Decagon + GNBR | None | --- | --- |
| GCN | None | GTex (kidney, bladder, breast) | --- | --- |
| GCN | PP-Decagon | GTex (kidney, bladder, breast) | --- | --- |
| GCN | PP-Decagon + GNBR | GTex (kidney, bladder, breast) | --- | --- |

*Using cancer gene labels from Network of Cancer Genes (*711 known cancer genes*)
