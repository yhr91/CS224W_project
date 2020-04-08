import pandas as pd
import Bio.SwissProt as sp
import ptm_keyword_parser
import gzip
import pickle
import re
from collections import defaultdict

class BioGrid:
    def __init__(self):
        self.df = pd.read_csv("./BIOGRID-PTMS-3.5.183.ptm/BIOGRID-PTM-3.5.183.ptmtab.txt", sep='\t')
        self.processed_df = self.get_features()
        self.PTMs = set()
        for item in self.processed_df["PTM"]:
            self.PTMs.add(item)

    def get_features(self):
        new_df = pd.DataFrame()
        new_df["Gene"] = self.df["Entrez Gene ID"]
        new_df["PTM"] = self.df["Post Translational Modification"]
        new_df["Position"] = self.df["Position"]
        new_df["Organism"] = self.df["Organism ID"]
        new_df = new_df[new_df["Organism"] == 9606].reset_index().drop("index", axis=1)
        return new_df

class UniProt:
    def __init__(self):
        self.no_keywords = set()
        self.ptm_keyword_dict = self.create_ptm_keyword_dict()
        self.ptm_set = set()
        self.feature_set = set()
        self.protein_feature_dict = dict()


    def create_ptm_keyword_dict(self):
        output_dict = dict()
        for record in ptm_keyword_parser.parse(
            open("/Users/benjaminangulo/Documents/Stanford2019/CS224W/CS224W_project/GNN_embedding"
            "/Protein_Features/ptmlist_04_08_20.txt", "r")):
            if record.get("KW", None) is None:
                # print(record)
                self.no_keywords.add(record["ID"])
                continue
            output_dict[record["ID"]] = record["KW"]
        return output_dict


    def parse_records(self):
        """A method to open and parse UniProt according to their keywords. Similar to what was
        done in Ata et al. 2018.

        :return: A
        """
        for record in sp.parse(gzip.open(
                "/Users/benjaminangulo/Documents/Stanford2019/CS224W"
                                 "/CS224W_project"
                   "/GNN_embedding"
               "/Protein_Features/human_uniprot_04_07_20.gz", 'rt')):
            # print(record.taxonomy_id)
            # if record.organism != "Homo sapiens":
            #     continue
            # print(record.features[0])
            # for comment in record.comments:
            #     if comment.startswith("SUBCELLULAR LOCATION"):
            #         print(comment)
            self.extract_features_to_dict(record)


    def extract_localization(self):
        pass

    def extract_features_to_dict(self, record):
        # print(record.accessions[0])
        features = defaultdict(int)
        for feature in record.features:
            self.feature_set.add(feature.type)
            if feature.type == "MOD_RES":
                # print(feature)
                modification = feature.qualifiers["note"].split(";")[0].rstrip(".")
                key_word_mod = self.ptm_keyword_dict.get(modification, modification)
                self.ptm_set.add(key_word_mod)
                features[key_word_mod] += 1

            if feature.type == "TRANSMEM":
                features[feature.type] += 1

            if feature.type == "INTRAMEM":
                features[feature.type] += 1

            if feature.type == "CROSSLNK":
                if re.search("NEDD", feature.qualifiers["note"], flags=re.I):
                    features["Neddylation"] += 1
                if re.search("ubiquitin", feature.qualifiers["note"], flags=re.I):
                    features["Ubiquitination"] += 1
                if re.search("SUMO", feature.qualifiers["note"], flags=re.I):
                    features["Sumoylation"] += 1

            if feature.type == "CARBOHYD":
                modification = feature.qualifiers["note"].split(";")[0].rstrip(".")
                key_word_mod = self.ptm_keyword_dict.get(modification, modification)
                self.ptm_set.add(key_word_mod)
                features[key_word_mod] += 1

            if feature.type == "LIPID":
                modification = feature.qualifiers["note"].split(";")[0].rstrip(".")
                key_word_mod = self.ptm_keyword_dict.get(modification, modification)
                self.ptm_set.add(key_word_mod)
                features[key_word_mod] += 1

            # if feature.type == "METAL":
            #     modification = feature.qualifiers["note"].split(";")[0]
            #     print(modification)

        self.protein_feature_dict[record.accessions[0]] = features

    def convert_dict_to_df(self):
        df = pd.DataFrame(self.protein_feature_dict).transpose()
        df.to_pickle("uniprot_ptm_features.pkl")
        print(df)
                # print(feature)
                # features[feature.type] += 1


#top_dom correlate which portion is extracellular or intra/etc.


if __name__ == '__main__':
    bg = BioGrid()
    print(bg.processed_df)
    print(bg.PTMs)
    up = UniProt()
    up.parse_records()
    up.convert_dict_to_df()
    # print(up.ptm_set)
    # print(up.no_keywords)
    # print(up.feature_set)