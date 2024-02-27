--convert flowers to hd5
import os
from os.path import join, isfile
import numpy as np
import h5py
from glob import glob
import torch
from PIL import Image
import yaml
import io
import pdb

images_path = '/content/drive/MyDrive/Colab Notebooks/Flowers/102flowers/jpg/'
embedding_path = '/content/file_pt/'
text_path = '/content/drive/MyDrive/Colab Notebooks/Flowers/cvpr2016_flowers/text_c10/'
datasetDir = '/content/flowers.hdf5'

val_classes = open('/content/drive/MyDrive/Colab Notebooks/Flowers/flowers_icml/flowers_icml/valclasses.txt').read().splitlines()
train_classes = open('/content/drive/MyDrive/Colab Notebooks/Flowers/flowers_icml/flowers_icml/trainclasses.txt').read().splitlines()
test_classes = open('/content/drive/MyDrive/Colab Notebooks/Flowers/flowers_icml/flowers_icml/testclasses.txt').read().splitlines()

f = h5py.File(datasetDir, 'w')
train = f.create_group('train')
valid = f.create_group('valid')
test = f.create_group('test')

for _class in sorted(os.listdir(embedding_path)):
    split = ''
    if _class in train_classes:
        split = train
    elif _class in val_classes:
        split = valid
    elif _class in test_classes:
        split = test

    data_path = os.path.join(embedding_path, _class)
    txt_path = os.path.join(text_path, _class)
    for example, txt_file in zip(sorted(glob(data_path + "/*.pt")), sorted(glob(txt_path + "/*.txt"))):
        example_data = torch.load(example)
        img_path = example_data['img']
        embeddings = example_data['txt']
        example_name = img_path[:-4]

        with open(txt_file, "r") as txt_f:
            txt = txt_f.readlines()

        img_path = os.path.join(images_path, img_path)
        img = open(img_path, 'rb').read()

        txt_choice = np.random.choice(range(10), 5)

        embeddings = embeddings[txt_choice]
        txt = np.array(txt)
        txt = txt[txt_choice]
        dt = h5py.special_dtype(vlen=str)

        for c, e in enumerate(embeddings):
            ex = split.create_group(example_name + '_' + str(c))
            ex.create_dataset('name', data=example_name)
            ex.create_dataset('img', data=np.void(img))
            ex.create_dataset('embeddings', data=e)
            ex.create_dataset('class', data=_class)
            ex.create_dataset('txt', data=txt[c].astype(object), dtype=dt)

        print(example_name, txt[1], _class)
