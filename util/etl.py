import os
import pickle
import random

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from skimage.io import imread, imsave
from skimage.transform import rotate

def onehot(t, num_classes):
    """Converts matrix of id numbers into a matrix of onehot vectors
    representing the same."""
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[int(row), int(col)] = 1
    return out

class load_data():
    """Loads data from csv and image directories.
    """
    def __init__(self, train_path, test_path, image_paths, image_shape=(128, 128), seed=42):
        """Reads csvs and loads images."""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        image_paths = image_paths
        image_shape = image_shape
        self._load(train_df, test_df, image_paths, image_shape)

    @staticmethod
    def _path_to_dict(image_paths):
        """Matches path to image with image number."""
        path_dict = dict()
        for image_path in image_paths:
            num_path = int(os.path.basename(image_path[:-4]))
            path_dict[num_path] = image_path
        return path_dict

    @staticmethod
    def _merge_image_df(df, path_dict):
        """Merges ids for images with image path."""
        split_path_dict = dict()
        for _, row in df.iterrows():
            split_path_dict[row['id']] = path_dict[row['id']]
        image_frame = pd.DataFrame(split_path_dict.values(), columns=['image'])
        df_image =  pd.concat([image_frame, df], axis=1)
        return df_image

    def _load(self, train_df, test_df, image_paths, image_shape):
        self.image_paths = image_paths
        path_dict = self._path_to_dict(self.image_paths) # numerate image paths and make it a dict
        # merge image paths with data frame

        self.train_image_df = self._merge_image_df(train_df, path_dict)
        self.test_image_df = self._merge_image_df(test_df, path_dict)
        self.le = LabelEncoder().fit(self.train_image_df['species'])
        # labels for train
        t_train = self.le.transform(self.train_image_df['species'])
        # getting data
        train_data = self._make_dataset(self.train_image_df, image_shape, t_train)
        test_data = self._make_dataset(self.test_image_df, image_shape)
        # need to reformat the train for validation split reasons in the batch_generator
        self.train = self._format_dataset(train_data, for_train=True)
        self.test = self._format_dataset(test_data, for_train=False)
        print "data loaded"


    def _make_dataset(self, df, image_shape, t_train=None):
        """Matches image with features and image identification."""
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
            sample['features'] = features
            if t_train is not None:
                sample['t'] = np.asarray(t_train[i], dtype='int32')
            image = imread(row['image'], as_grey=True)
            #image = helpers.scale_resize(image, (1706, 1706), image_shape)
            image = np.expand_dims(image, axis=2)
            #image = os.path.splitext(os.path.basename(row['image']))[0]
            sample['image'] = image
            data[row['id']] = sample
        return data

    @staticmethod
    def _format_dataset(df, for_train):
        """Merge all features."""
        data = dict()
        value = df.values()[0]
        img_tot_shp = tuple([len(df)] + list(value['image'].shape))
        data['images'] = np.zeros(img_tot_shp, dtype='float32')
        #data['images'] = np.zeros((len(df),), dtype='object')
        #data['margins'] = np.zeros(feature_tot_shp, dtype='float32')
        #data['shapes'] = np.zeros(feature_tot_shp, dtype='float32')
        #data['textures'] = np.zeros(feature_tot_shp, dtype='float32')
        data['features'] = np.zeros((len(df), 64 * 3), dtype='float32')
        if for_train:
            data['ts'] = np.zeros((len(df),), dtype='int32')
        else:
            data['ids'] = np.zeros((len(df),), dtype='int32')
        for i, pair in enumerate(df.items()):
            key, value = pair
            #data['images'][i] = '%s/%s%s' % (os.path.dirname(self.image_paths[0]), value['image'], os.path.splitext(self.image_paths[0])[1])
            data['images'][i] = value['image']
            data['features'][i] = value['features']
            if for_train:
                data['ts'][i] = value['t']
            else:
                data['ids'][i] = key
        return data

    def _augment_dataset(df):
        """Augments images.  Typically not useful for this object."""
        new_dataset = []
        id_start = df['id'].max() + 1
        counter = 0
        for _, row in df.iterrows():
            image = imread(row['image'], as_grey=True)
            for angle in [90, 180, 270, 'LR', 'UD']:
                new_row = dict()
                new_row['id'] = id_start + counter
                new_row['species'] = row['species']
                new_image = '%s/%s%s' % (os.path.dirname(row['image']), new_row['id'], os.path.splitext(row['image'])[1])
                new_row['image'] = new_image
                new_dataset.append(new_row)
                if isinstance(angle, int):
                    imsave(new_image, rotate(image, angle, resize=False))
                elif angle == 'LR':
                    imsave(new_image, np.fliplr(image))
                elif angle == 'UD':
                    imsave(new_image, np.flipud(image))
                counter += 1
            print 'augmented: ', row['image']
        new_df = pd.concat((df, pd.DataFrame(new_dataset)))
        new_df.to_csv('./data/train_images.csv')
        return new_df


class batch_generator():
    """Generates batches for training, validation, and testing."""
    def __init__(self, train, test, batch_size=64, num_classes=99,
                 num_iterations=5e3, num_features=64*3, seed=42, train_size=0.25, val_size=0.1, class_size=1.0):
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
            self._classes = sorted(set(self._train['ts']))[:self._num_classes]
        else:
            self._classes = set(self._train['ts'])


        # idcs that we care about for train
        self._idcs_train = [idx for idx, class_id in enumerate(self._train['ts']) if class_id in self._classes]
        self._train_ids = [self._train['ts'][i] for i in self._idcs_train]

        self._idcs_train, self._train_ids, self._idcs_valid, self._valid_ids = self._split(self._train_ids, self._idcs_train, self._val_size)
#

        self._train_size = train_size
        if self._train_size < 1.0:
            self._idcs_train, self._train_ids, _, _ = self._split(self._train_ids, self._idcs_train, self._train_size)

    def _split(self, ids, idcs_id, portion):
        print portion
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
        #batch_holder['margins'] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        #batch_holder['shapes'] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        #batch_holder['textures'] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        #batch_holder['images'] = np.zeros(tuple([self._batch_size] + self._image_shape), dtype='float32')
        batch_holder['features'] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        if (purpose == "train") or (purpose == "valid"):
            batch_holder['images'] = np.zeros(tuple([self._batch_size] + self._image_shape), dtype='object')
            batch_holder['ts'] = np.zeros((self._batch_size, self._num_classes), dtype='float32')
        else:
            batch_holder['images'] = []
            batch_holder['ids'] = []
        return batch_holder


    def gen_valid(self):
        batch = self._batch_init(purpose='train')
        i = 0
        for idx in self._idcs_valid:
            #batch['margins'][i] = self._train['margins'][idx]
            #batch['shapes'][i] = self._train['shapes'][idx]
            #batch['textures'][i] = self._train['textures'][idx]
            batch['images'][i] = self._train['images'][idx]
            batch['features'][i] = self._train['features'][idx]
            #batch['features'][i] = np.concatenate((self._train['margins'][idx], self._train['shapes'][idx], self._train['textures'][idx]))
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
            #batch['margins'][i] = self._test['margins'][idx]
            #batch['shapes'][i] = self._test['shapes'][idx]
            #batch['textures'][i] = self._test['textures'][idx]
            batch['images'].append(self._test['images'][idx])
            batch['ids'].append(self._test['ids'][idx])
            batch['features'][i] = self._test['features'][idx]
            #batch['features'][i] = np.concatenate((self._test['margins'][idx], self._test['shapes'][idx], self._test['textures'][idx]))
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
                #batch['margins'][i] = self._train['margins'][idx]
                #batch['shapes'][i] = self._train['shapes'][idx]
                #batch['textures'][i] = self._train['textures'][idx]
                batch['images'][i] = self._train['images'][idx]
                batch['ts'][i] = onehot(np.asarray([self._train['ts'][idx]], dtype='float32'), self._num_classes)
                batch['features'][i] = self._train['features'][idx]
                #batch['features'][i] = np.concatenate((self._train['margins'][idx], self._train['shapes'][idx], self._train['textures'][idx]))
                i += 1
                if i >= self._batch_size:
                    yield batch
                    batch = self._batch_init(purpose='train')
                    i = 0
                    iteration += 1
                    if iteration >= self._num_iterations:
                        break

