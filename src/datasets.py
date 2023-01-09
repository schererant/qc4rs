from json import load
from urllib.request import urlretrieve
from scipy.io import loadmat
from sklearn.decomposition import PCA
import sklearn.model_selection
from sklearn import preprocessing
from scipy import ndimage
import numpy as np
import os

from tqdm import tqdm

#TODO: save and read DATASETS_CONFIG in/from a json file
DATASETS_CONFIG = {
    'PaviaC': {
        'urls': ['http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat', 
                    'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'],
        'img': 'Pavia.mat',
        'gt': 'Pavia_gt.mat',
        'img_key': 'pavia',
        'gt_key': 'pavia_gt',
        'rgb': (55, 41, 12),
        'label_values': ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"],
        'ignored_labels': [0]
        },
    'PaviaU': {
        'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                    'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
        'img': 'PaviaU.mat',
        'gt': 'PaviaU_gt.mat',
        'img_key': 'paviaU',
        'gt_key': 'paviaU_gt',
        'rgb': (55, 41, 12),
        'label_values': ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows'],
        'ignored_labels': [0]
        }
}


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_dataset(dataset_name, target_folder, dataset_config=DATASETS_CONFIG):
    for url in dataset_config[dataset_name]['urls']:
        # download the files
        filename = url.split('/')[-1]
        if not os.path.exists(target_folder + '/' + filename):
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                        desc="Downloading {}".format(filename)) as t:
                urlretrieve(url, filename=target_folder + "/" + filename,
                                    reporthook=t.update_to)


# Taken from hyperpsectral pacakge
def sample_gt(gt, train_size, mode='random', random_state=None):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
    
    if mode == 'random':
       train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y, random_state=random_state)
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
       test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    elif mode == 'fixed':
       print("Sampling {} with train size = {}".format(mode, train_size))
       train_indices, test_indices = [], []
       for c in np.unique(gt):
           if c == 0:
              continue
           indices = np.nonzero(gt == c)
           X = list(zip(*indices)) # x,y features

           train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
           train_indices += train
           test_indices += test
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


class Dataset:
    def __init__(self, dataset_name: str, target_folder: str, pca_components=None, datasets=DATASETS_CONFIG):
        self.img = loadmat(target_folder + '/' + datasets[dataset_name]['img'])[datasets[dataset_name]['img_key']]
        self.gt = loadmat(target_folder + '/' + datasets[dataset_name]['gt'])[datasets[dataset_name]['gt_key']]
        self.img_pca = None
        self.rgb = datasets[dataset_name]['rgb']
        self.label_values = datasets[dataset_name]['label_values']
        self.ignored_labels = datasets[dataset_name]['ignored_labels']
        self.pca_components = pca_components

        if pca_components is not None:
            print('PCA reduction to {} components...'.format(pca_components))
            self.pca_reduce(pca_components)

        # Preprocess data
        print('Filtering NaN...')
        self.filter_NaN()
        print('Normalizing...')
        self.normalize()


    def filter_NaN(self):

        # Filter NaN out and replace with 0
        nan_mask = np.isnan(self.img.sum(axis=-1))
        if np.count_nonzero(nan_mask) > 0:
            print("Warning: NaN have been found and replaced by 0.")
        self.img[nan_mask] = 0
        self.gt[nan_mask] = 0
        self.ignored_labels.append(0)

    def normalize(self):

        # For img 
        self.img = np.asarray(self.img, dtype='float32')
        features_reshaped = self.img.reshape(np.prod(self.img.shape[:2]), np.prod(self.img.shape[2:]))
        features_norm = preprocessing.minmax_scale(features_reshaped)
        self.img = features_norm.reshape(self.img.shape)

        if self.pca_components is not None:
            # For pca
            self.img_pca = np.asarray(self.img_pca, dtype='float32')
            self.img_pca = preprocessing.minmax_scale(self.img_pca)
            self.img_pca = self.img_pca.reshape(self.img.shape[0], self.img.shape[1], self.pca_components)


    #TODO: What complexity does it add? https://scicomp.stackexchange.com/questions/3220/fastest-pca-algorithm-for-high-dimensional-data
    #TODO: Write about potential ways to do it quantum
    def pca_reduce(self, n_components=None):
        if n_components is None:
            n_components = self.img.shape[-1]

        pca = PCA(n_components=n_components)
        self.img_pca = np.reshape(self.img, (self.img.shape[0] * self.img.shape[1], self.img.shape[2]))
        pca.fit(self.img_pca)
        self.img_pca = pca.transform(self.img_pca)

    
    # def apply_filter(self, filter):
    #     if filter == 'median':
    #         self.img = ndimage.median_filter(self.img, size=3)
    #     elif filter == 'gaussian':
    #         self.img = ndimage.gaussian_filter(self.img, sigma=1)
    #     elif filter == 'uniform':
    #         self.img = ndimage.uniform_filter(self.img, size=3)
    #     elif filter == 'bilateral':


        