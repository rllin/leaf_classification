import os
import subprocess
import itertools
from datetime import datetime
import time
import glob
import random
import pickle
from collections import Counter

import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from skimage.io import imread
from skimage.transform import resize

import helpers

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[int(row), int(col)] = 1
    return out


class load_data():
    # data_train, data_test and le are public
    def __init__(self, train_path, test_path, image_paths, image_shape=(128, 128), seed=42, verbose_flag=False):
        self._seed = seed
        random.seed(self._seed)

        self._verbose_flag = verbose_flag
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        image_paths = image_paths
        image_shape = image_shape
        self._load(train_df, test_df, image_paths, image_shape)

    '''
    def _frac(self, data):

        inds = []
        for k, v in data.groupby('species').indices.items():
            num_take = min(2, int(round(self._data_frac * len(v))))
            inds += random.sample(v, num_take)
        return data.ix[random.sample(data.index, int(round(self._data_frac * len(data))))]
    '''

    def _load(self, train_df, test_df, image_paths, image_shape):
        print "loading data ..."
        # load train.csv
        path_dict = self._path_to_dict(image_paths) # numerate image paths and make it a dict
        # merge image paths with data frame

        train_image_df = self._merge_image_df(train_df, path_dict)
        test_image_df = self._merge_image_df(test_df, path_dict)
        # label encoder-decoder (self. because we need it later)
        self.le = LabelEncoder().fit(train_image_df['species'])
        # labels for train
        t_train = self.le.transform(train_image_df['species'])
        # getting data
        train_data = self._make_dataset(train_image_df, image_shape, t_train)
        test_data = self._make_dataset(test_image_df, image_shape)
        # need to reformat the train for validation split reasons in the batch_generator
        self.train = self._format_dataset(train_data, for_train=True)
        self.test = self._format_dataset(test_data, for_train=False)
        print "data loaded"


    def _path_to_dict(self, image_paths):
        path_dict = dict()
        for image_path in image_paths:
            num_path = int(os.path.basename(image_path[:-4]))
            path_dict[num_path] = image_path
        return path_dict

    def _merge_image_df(self, df, path_dict):
        split_path_dict = dict()
        for index, row in df.iterrows():
            split_path_dict[row['id']] = path_dict[row['id']]
        image_frame = pd.DataFrame(split_path_dict.values(), columns=['image'])
        df_image =  pd.concat([image_frame, df], axis=1)
        return df_image


    def _make_dataset(self, df, image_shape, t_train=None):
        if t_train is not None:
            print "loading train ..."
        else:
            print "loading test ..."
        # make dataset
        data = dict()
        # merge image with 3x64 features
        for i, dat in enumerate(df.iterrows()):
            index, row = dat
            sample = dict()
            if t_train is not None:
                features = row.drop(['id', 'species', 'image'], axis=0).values
            else:
                features = row.drop(['id', 'image'], axis=0).values
            sample['margin'] = features[:64]
            sample['shape'] = features[64:128]
            sample['texture'] = features[128:]
            if t_train is not None:
                sample['t'] = np.asarray(t_train[i], dtype='int32')
            image = imread(row['image'], as_grey=True)
            image = helpers.scale_resize(image, (1706, 1706), image_shape)
            image = np.expand_dims(image, axis=2)
            sample['image'] = image
            data[row['id']] = sample
            if i % 100 == 0 and self._verbose_flag:
                print "\t%d of %d" % (i, len(df))
        return data

    def _format_dataset(self, df, for_train):
        # making arrays with all data in, is nessesary when doing validation split
        data = dict()
        value = df.values()[0]
        img_tot_shp = tuple([len(df)] + list(value['image'].shape))
        data['images'] = np.zeros(img_tot_shp, dtype='float32')
        feature_tot_shp = (len(df), 64)
        data['margins'] = np.zeros(feature_tot_shp, dtype='float32')
        data['shapes'] = np.zeros(feature_tot_shp, dtype='float32')
        data['textures'] = np.zeros(feature_tot_shp, dtype='float32')
        if for_train:
            data['ts'] = np.zeros((len(df),), dtype='int32')
        else:
            data['ids'] = np.zeros((len(df),), dtype='int32')
        for i, pair in enumerate(df.items()):
            key, value = pair
            data['images'][i] = value['image']
            data['margins'][i] = value['margin']
            data['shapes'][i] = value['shape']
            data['textures'][i] = value['texture']
            if for_train:
                data['ts'][i] = value['t']
            else:
                data['ids'][i] = key
        return data


class batch_generator():
    def __init__(self, train, test, batch_size=64, num_classes=99,
                 num_iterations=5e3, num_features=64, seed=42, train_size=0.25, val_size=0.1, class_size=1.0):
        print "initiating batch generator with (%d/%d) classes and %f of the training data and %f as validation" % (int(round(class_size * num_classes)), num_classes, train_size, val_size)

        num_classes = int(round(class_size * num_classes))
        self._seed = seed

        self._train = train
        self._test = test

        # get image size
        value = self._train['images'][0]
        self._image_shape = list(value.shape)
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._num_iterations = num_iterations
        self._num_features = num_features
        self._val_size = val_size

        if class_size < 1.0:
            #self._classes = random.sample(set(self._train['ts']), self._num_classes)
            self._classes = sorted(set(self._train['ts']))[:self._num_classes]
        else:
            self._classes = set(self._train['ts'])

        #print 'original classes: ', Counter(self._train['ts'])
        #print 'smaller classes: ', self._classes

        # idcs that we care about for train
        self._idcs_train = [idx for idx, id in enumerate(self._train['ts']) if id in self._classes]
        self._train_ids = [self._train['ts'][i] for i in self._idcs_train]
        #print 'train: ', Counter(self._train_ids)

        self._idcs_train, self._train_ids, self._idcs_valid, self._valid_ids = self._split(self._train_ids, self._idcs_train, self._val_size)
#
        #print 'train (split valid): ', Counter(self._train_ids), Counter(self._valid_ids)

        self._train_size = train_size
        if self._train_size < 1.0:
            self._idcs_train, self._train_ids, _, _ = self._split(self._train_ids, self._idcs_train, self._train_size)

        '''
        print self._train_size
        print 'train (make smaller): ', Counter(self._train_ids)

        print "batch generator initiated ..."

        print 'train'
        print self._idcs_train
        print self._train_ids
        print [self._train['ts'][idx] for idx in self._idcs_train]

        print 'valid'
        print self._idcs_valid
        print self._valid_ids
        print [self._train['ts'][idx] for idx in self._idcs_valid]
        '''

    def _split(self, ids, idcs_id, portion):
        idcs_train, idcs_valid = iter(
            StratifiedShuffleSplit(ids,
                                   n_iter=1,
                                   test_size=portion,
                                   random_state=self._seed)).next()
        return [idcs_id[i] for i in idcs_train], [ids[i] for i in idcs_train], [idcs_id[i] for i in idcs_valid], [ids[i] for i in idcs_valid]

    def _shuffle_train(self):
        np.random.shuffle(self._idcs_train)

    def _batch_init(self, purpose):
        assert purpose in ['train', 'valid', 'test']
        batch_holder = dict()
        batch_holder['margins'] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder['shapes'] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder['textures'] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder['images'] = np.zeros(tuple([self._batch_size] + self._image_shape), dtype='float32')
        batch_holder['flat_images'] = np.zeros((self._batch_size, np.prod(self._image_shape)), dtype='float32')
        batch_holder['features'] = np.zeros((self._batch_size, self._num_features * 3), dtype='float32')
        if (purpose == "train") or (purpose == "valid"):
            batch_holder['ts'] = np.zeros((self._batch_size, self._num_classes), dtype='float32')
        else:
            batch_holder['ids'] = []
        return batch_holder


    def gen_valid(self):
        batch = self._batch_init(purpose='train')
        i = 0
        for idx in self._idcs_valid:
            batch['margins'][i] = self._train['margins'][idx]
            batch['shapes'][i] = self._train['shapes'][idx]
            batch['textures'][i] = self._train['textures'][idx]
            batch['images'][i] = self._train['images'][idx]
            batch['flat_images'][i] = self._train['images'][idx].flatten()
            batch['features'][i] = np.concatenate((self._train['margins'][idx], self._train['shapes'][idx], self._train['textures'][idx]))
            batch['ts'][i] = onehot(np.asarray([self._train['ts'][idx]], dtype='float32'), self._num_classes)
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self._batch_init(purpose='valid')
                i = 0
        if i != 0:
            yield batch, i

    def gen_test(self):
        batch = self._batch_init(purpose='test')
        i = 0
        for idx in range(len(self._test['ids'])):
            batch['margins'][i] = self._test['margins'][idx]
            batch['shapes'][i] = self._test['shapes'][idx]
            batch['textures'][i] = self._test['textures'][idx]
            batch['images'][i] = self._test['images'][idx]
            batch['flat_images'][i] = self._test['images'][idx].flatten()
            batch['ids'].append(self._test['ids'][idx])
            batch['features'][i] = np.concatenate((self._test['margins'][idx], self._test['shapes'][idx], self._test['textures'][idx]))
            #batch['ids'].append(onehot(np.asarray([self._test['ids'][idx]], dtype='float32'), self._num_classes))
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self._batch_init(purpose='test')
                i = 0
        if i != 0:
            yield batch, i


    def gen_train(self):
        batch = self._batch_init(purpose='train')
        iteration = 0
        i = 0
        while True:
            # shuffling all batches
            self._shuffle_train()
            for idx in self._idcs_train:
                # extract data from dict
                batch['margins'][i] = self._train['margins'][idx]
                batch['shapes'][i] = self._train['shapes'][idx]
                batch['textures'][i] = self._train['textures'][idx]
                batch['images'][i] = self._train['images'][idx]
                batch['flat_images'][i] = self._train['images'][idx].flatten()
                batch['ts'][i] = onehot(np.asarray([self._train['ts'][idx]], dtype='float32'), self._num_classes)
                batch['features'][i] = np.concatenate((self._train['margins'][idx], self._train['shapes'][idx], self._train['textures'][idx]))
                i += 1
                if i >= self._batch_size:
                    yield batch
                    batch = self._batch_init(purpose='train')
                    i = 0
                    iteration += 1
                    if iteration >= self._num_iterations:
                        break



