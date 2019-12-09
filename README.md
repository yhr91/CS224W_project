## Graph embeddings of enriched protein-protein interaction (PPI) networks for identification of disease nodes

Goals:
Integration of traditionally distinct data modalities (including literature based knowledge graphs and multi-omics datasets) to enrich the information content of a PPI graph
Developing new methods for embedding these enriched graphs within a shared space that allows for better detection of disease modules in the graph



### Disease node classification leaderboard

| Model | Graph | Features | MRR | F1* |
| --- | --- | --- | --- | --- |
| Diamond | PP-Decagon | None | 0.11 | --- |
| GCN | PP-Decagon | None | --- | --- |
| GCN | PP-Decagon | GTex | --- | 0.68 |
| GCN | PP-Decagon | UniProt | --- | --- |
| GCN | PP-Decagon + GNBR | None | --- | --- |
| GCN | PP-Decagon + GNBR | GTex+ UniProt | --- | --- |

*Using cancer gene labels from Network of Cancer Genes (*711 known cancer genes*)
Results produced after applying 5-fold cros validation and choosing the best performing model across all folds
