"""
python3 datasets/utils/FELICE_build_annotations.py
"""

import os
import csv
import pandas
from tqdm import tqdm

from commands import DEMO7


IN_DATASET = "MIVIA_HGR"
IN_ANNOTATION_FILE = "hgr_custom_dataset_rgb.csv"
OUT_DATASET = "MIVIA_HGR"
OUT_ANNOTATION_FILE = "hgr_custom_dataset_rgb_DEMO7.csv"
IN_ANNOTATIONS_PATH = os.path.join("datasets", IN_DATASET, IN_ANNOTATION_FILE)
OUT_ANNOTATIONS_PATH = os.path.join("datasets", OUT_DATASET, OUT_ANNOTATION_FILE)


'''Read input annotation file'''
print(IN_ANNOTATIONS_PATH)
in_df = pandas.read_csv(filepath_or_buffer=IN_ANNOTATIONS_PATH, sep=';')
columns = in_df.columns

'''Build out dictionary with the same structure of the input annotation dataframe'''
out_dict =dict()
for column in columns:
    out_dict[column] = list()

'''Iterate on each row to change the IDs'''
row_iter = tqdm(range(len(in_df)))
for i in row_iter:
    row = in_df.iloc[i]
    for column in columns:
        if column == "class_name":
            old_labels = row[column].replace('[', '').replace(']','').split(", ")
            labels = list()
            for old_label in old_labels:
                labels.append(DEMO7[int(old_label)])
            out_dict[column].append(labels)
        else:
            out_dict[column].append(row[column])

'''Save the out annotation file with update IDs'''
out_df = pandas.DataFrame(data=out_dict, columns=columns)
out_df.to_csv(path_or_buf=OUT_ANNOTATIONS_PATH, index=False, sep=';')