import pandas as pd
import os
from shutil import copyfile


os.mkdir('datasets/nature')
table = pd.read_csv('beauty-icwsm15-dataset.tsv', sep='\t')
selected = table[table['category'] == 'nature']
for id in selected['#flickr_photo_id']:
    copyfile(os.path.join('~/Flickr15k/imgs', id + '.jpg'), os.path.join('datasets/nature', id + '.jpg'))
