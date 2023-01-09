# Docstring
import pickle
import time
import warnings
from featuremaps import FeatureMap
from kernels import QuantumKernel
from pennylane import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from utils import *

warnings.simplefilter('ignore', UserWarning) # Ignore Jax multi_dispatch pennylane warning


def get_model():
    pass


class QuantumSVM:
    def __init__(self, featuremap_name, kernel_name):
        self.featuremap = FeatureMap(featuremap_name).get_feature_map()
        self.kernel = QuantumKernel(kernel_name).get_quantumkernel()
        self.train_results = {'featuremap_name': featuremap_name, 'kernel_name': kernel_name}
        self.test_results = {'featuremap_name': featuremap_name, 'kernel_name': kernel_name}
        self.gram_matrix = None
        self.predictions = None

    # Using precomputed kernels https://peetali.rbind.io/post/precomputed_kernel/
    def train_svm(self, train_data, train_gt, label_names, test_data=None, load_model=False, save_model=False):
        
        if not load_model:
            # Precompute Gram matrix
            print('Precomputing Gram matrix...')
            gram_matrix = self.kernel(feature_map=self.featuremap, train_data=train_data, test_data=test_data)
            
            # Initialize support vector classiifer
            svm = SVC(kernel="precomputed")

            # Train SVM
            svm.fit(gram_matrix, train_gt)
            #TODO: save model in respective folder for dataset and parameters
            if save_model:
                pickle.dump(svm, open('../out/saved_models/QuantumSVM.pkl', 'wb'))
                np.save('../out/saved_models/QuantumSVM_gram_matrix.npy', gram_matrix)
                print('Model saved')
        else:
            try:
                print('Loading model...')
                svm = pickle.load(open('../out/saved_models/QuantumSVM.pkl', 'rb'))
                gram_matrix = np.load('../out/saved_models/QuantumSVM_gram_matrix.npy')
            except FileNotFoundError:
                print('Model does not exist. Please train and save model first.')
                quit()

        # Evaluate model
        self.train_results['predictions'] = svm.predict(gram_matrix)
        self.train_results["accuracy"] = np.sum(train_gt == self.train_results['predictions']) / len(train_gt)
        self.train_results['confusion_matrix'] = confusion_matrix(train_gt, self.train_results['predictions'])
        self.train_results['classification_report'] = classification_report(train_gt, self.train_results['predictions'], target_names=label_names)

        # Accuracy on train set
        try:
            print_results_terminal(self.train_results)
        except ValueError:
            print('Saved model does not match current parameters. Please train and save model first.')

    # def train_kernel():
    #     pass

    #TODO: change Gram matrix to jax array
    def test(self, test_data, test_gt, train_data, label_names, load_model=True):
        print('Loading model...')
        svm = pickle.load(open('../out/saved_models/QuantumSVM.pkl', 'rb'))

        # compute gram matrix for testing 
        gram_start = time.time()
        if not load_model:
            gram_matrix_test = self.kernel(feature_map=self.featuremap, train_data=train_data, test_data=test_data) 
            np.save('../out/saved_models/QuantumSVM_gram_matrix_test.npy', gram_matrix_test)
            gram_end = time.time()
            print(f'Computing testing kernel took {gram_end - gram_start}')
        else:
            gram_matrix_test = np.load('../out/saved_models/QuantumSVM_gram_matrix_test.npy')
            print(gram_matrix_test[0])
            gram_end = time.time()
            print(f'Loading testing kernel took {gram_end - gram_start}')

        # Accuracy on test set with precomputed matrix
        test_start = time.time()
        pred_test = svm.predict(gram_matrix_test)
        test_end = time.time()
        print(f'svc precomputed kernel testing took {test_end - test_start}')
        print(f'Total time for testing: {test_end - gram_start}')

        # Evaluating results
        self.test_results['predictions'] = pred_test
        self.test_results["accuracy"] = np.sum(test_gt == self.test_results['predictions']) / len(test_gt)
        self.test_results['confusion_matrix'] = confusion_matrix(test_gt, self.test_results['predictions'])
        self.test_results['classification_report'] = classification_report(test_gt, self.test_results['predictions'], target_names=label_names)
        
        
        try:
            print_results_terminal(self.test_results)
        except ValueError:
            print('Saved model does not match current parameters. Please train and save model first.')

    # # Metrix
    # def calculate_frobenius_inner_product(self, A, B, normalized=False):
    #     """
    #     Calculate the Frobenius inner product of two matrices.
    #     Args:
    #         A: first matrix (numpy ndarray)
    #         B: second matrix (numpy nodarray)
    #         normalized: if True the result is divided by its norm
    #     Returns:
    #         Frobenius inner product (float)
    #     """
    #     norm = np.sqrt(np.sum(A * A) * np.sum(B * B)) if normalized else 1
    #     return np.sum(A * B) / norm


    # # Alignment
    # def calculate_kernel_target_alignment(self, gram_matrix, label_vector):
    #     """
    #     Calculate the kernel-target alignment.
    #     Args:
    #         gram_matrix: data points (square) Gram matrix (numpy ndarray)
    #         label_vector: label vector (numpy ndarray)
    #     Return:
    #         Kernel-target alignment (float).
    #     """
    #     Y = np.outer(label_vector, label_vector)
    #     return self.calculate_frobenius_inner_product(gram_matrix, Y, normalized=True)

    # # SVM
    # @staticmethod
    # def calculate_generalization_accuracy(training_gram, training_labels, testing_gram, testing_labels):
    #     """
    #     Calculate accuracy wrt a precomputed kernel, a training and testing set
    #     Args:
    #         training_gram: Gram matrix of the training set, must have shape (N,N)
    #         training_labels: Labels of the training set, must have shape (N,)
    #         testing_gram: Gram matrix of the testing set, must have shape (M,N)
    #         testing_labels: Labels of the training set, must have shape (M,)
    #     Returns:
    #         generalization accuracy (float)
    #     """
    #     svm = SVC(kernel="precomputed")
    #     svm.fit(training_gram, training_labels)
    #     y_predict = svm.predict(testing_gram)
    #     correct = np.sum(testing_labels == y_predict)
    #     accuracy = correct / len(testing_labels)
    #     return accuracy

