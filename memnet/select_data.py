import pandas as pd
import os
from shutil import copyfile

dataset_dir = '/unreliable/aliaksandr/memorability/mem-optimization/datasets/nature'
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)
table = pd.read_csv('/unreliable/DATASETS/Aesthetic/Flickr15K/beauty-icwsm15-dataset.tsv', sep='\t')
selected = table[table['category'] == 'nature']
for id in selected['#flickr_photo_id']:
    file_name = os.path.join('/unreliable/DATASETS/Aesthetic/Flickr15K/imgs', str(id) + '.jpg')
    if os.path.exists(file_name):
        copyfile(file_name, os.path.join(dataset_dir, str(id) + '.jpg'))
