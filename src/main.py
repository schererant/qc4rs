""" Main module """
import argparse
import visdom
from datasets import *
from utils import *
from models import *


###############################################################################
# Define command line arguments
###############################################################################

datasets = ['PaviU']

# Define main arguments for parser
parser = argparse.ArgumentParser(
    description="Hyperspectral Image Classifier using Quantum Algortithms")
parser.add_argument("--dataset", type=str, default=None, help=datasets)
parser.add_argument("--model", type=str, default=None, help="Model to use")
parser.add_argument("--cuda", type=int, default=-1, help="CUDA device to use, -1 is CPU")
parser.add_argument('--folder',
                    type=str,
                    help="Folder where to store the datasets "
                         "(defaults to the current working directory).",
                    default="./Datasets/")
parser.add_argument('--restore', type=str, default=None,
    help="Weights to use for initialization, e.g. a checkpoint")
parser.add_argument('--explore_spectra', action='store_true')
parser.add_argument('--visdom', action='store_true', default=True)

# Models options
parser.add_argument('--train_size', type=float, default=0.2)
parser.add_argument('--sample_size', type=float, default=None)
# TODO: no pca components produces error
parser.add_argument('--pca_components', type=int, default=None)
parser.add_argument('--sample_mode', type=str, default='random')

# Quantum options


# Dataset options
group_dataset = parser.add_argument_group('Dataset options')

args = parser.parse_args()

###############################################################################
# Initialize parameters
###############################################################################

# Cuda device
# CUDA_DEVICE = get_device(args.cuda)
# Dataset
DATASET = args.dataset
# Model
MODEL = args.model
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Show spectra of labels in the dataset
SHOW_SPECTRA = args.explore_spectra
# Sample size
SAMPLE_SIZE = args.sample_size
# Train size
TRAIN_SIZE = args.train_size
# PCA components
PCA_COMPONENTS = args.pca_components
# Sample mode
SAMPLE_MODE = args.sample_mode
# Random State
RANDOM_STATE = 42
# Visdom
VISDOM = args.visdom
# Script path
SCRIPT_DIR = os.path.dirname(__file__)  # <-- absolute dir the script is in
REL_PATH = "../data"


def main():

    ###############################################################################
    # Initialize visdom
    ###############################################################################

    if VISDOM:
        vis = visdom.Visdom(env=DATASET + ' ' + MODEL)
        if not vis.check_connection:
            print("Visdom is not connected. Did you run 'python -m visdom.server' ?")
    else:
        vis = None

    ###############################################################################
    # Initialize dataset
    ###############################################################################
    # Check whether the dataset is already downloaded

    # target_folder
    target_folder = os.path.join(SCRIPT_DIR, REL_PATH)

    if not os.path.isdir(target_folder+'/'+DATASET):
        os.mkdir(target_folder+'/'+DATASET)
        print('Downloading dataset '+DATASET)
        download_dataset(DATASET, target_folder)

    dataset = Dataset(DATASET, target_folder, pca_components=PCA_COMPONENTS)

    # Display rgb image and ground truth in visdom
    show_img_gt(
        img=dataset.img,
        gt=dataset.gt,
        rgb=dataset.rgb,
        label_values=dataset.label_values,
        vis=vis
        )

    if SHOW_SPECTRA:
        explore_spectra(dataset.img, dataset.gt, dataset.label_values, vis)

    ###############################################################################
    # Sample data
    ###############################################################################

    train_gt, test_gt = sample_gt(dataset.gt, TRAIN_SIZE, SAMPLE_MODE, random_state=RANDOM_STATE)
    display_train_test_split(
        dataset.gt,
        train_gt,
        test_gt,
        "Training Set",
        "Test Set",
        dataset.label_values, vis
        )

    ###############################################################################
    # Initialize model
    ###############################################################################

    if MODEL == 'SVM':

        x_train, y_train, _ = build_dataset(
            img=dataset.img,
            gt=train_gt,
            ignored_labels=dataset.ignored_labels)
        # Assert of PCA_COMPONENTS are not None

        assert PCA_COMPONENTS is not None, "PCA_COMPONENTS must be specified"
        x_pca, _, _ = build_dataset(dataset.img_pca,
                                    train_gt,
                                    ignored_labels=dataset.ignored_labels)

        svm = sklearn.svm.SVC(kernel='rbf', C=1e3, gamma=0.1)
        svm.fit(x_pca, y_train)
        print('SVM model trained')
        print('Accuracy: ', svm.score(x_pca, y_train))
        print(classification_report(y_train, svm.predict(x_pca)))

        x_test, y_test, _ = build_dataset(
            img=dataset.img,
            gt=test_gt,
            ignored_labels=dataset.ignored_labels)

        print("Testing SVM model")
        x_test_pca, _, _ = build_dataset(dataset.img_pca,
                                         test_gt,
                                         ignored_labels=dataset.ignored_labels)
        print('Accuracy: ', svm.score(x_test_pca, y_test))
        print(classification_report(y_test, svm.predict(x_test_pca)))



    #TODO: Should we normalize the data between 0 and pi?

    elif MODEL == 'precomputed-SVM':
        # Training
        x_pca, y_train, _ = build_dataset(dataset.img_pca,
                                          train_gt,
                                          ignored_labels=dataset.ignored_labels)
        svm = sklearn.svm.SVC(kernel='precomputed')
        gram_matrix = np.dot(x_pca, x_pca.T)
        svm.fit(gram_matrix, y_train)
        print('SVM model trained')
        print('Accuracy: ', svm.score(gram_matrix, y_train))
        print(classification_report(y_train, svm.predict(gram_matrix)))

        # Testing
        x_test_pca, y_test, _ = build_dataset(dataset.img_pca,
                                              test_gt,
                                              ignored_labels=dataset.ignored_labels)
        #TODO: Why is gram matrix classically just half the size?
        gram_matrix_test = np.dot(x_test_pca, x_pca.T)
        # np.save('gram_matrix_test.npy', gram_matrix_test)


        print('Accuracy on test set: ', svm.score(gram_matrix_test, y_test))
        print(classification_report(y_test, svm.predict(gram_matrix_test)))

    elif MODEL == 'quantum-SVM':
        print('Selected Model: Quantum-SVM')

        # Inititalize dataset
        x_train, y_train, train_indices = build_dataset(dataset.img,
                                                        train_gt,
                                                        ignored_labels=dataset.ignored_labels)
        x_pca, _, _ = build_dataset(dataset.img_pca,
                                    train_gt,
                                    ignored_labels=dataset.ignored_labels)
        #input_summary(x_train)

        # Initialize model
        quantum_svm = QuantumSVM(featuremap_name='zz_entangled',
                                kernel_name='quantum_kernel')

        # Train model
        quantum_svm.train_svm(
            train_data=x_pca,
            train_gt=y_train,
            label_names=[j for i, j in enumerate(dataset.label_values)
                         if i not in dataset.ignored_labels], load_model=True)
        pretty_print_confusion_matrix(quantum_svm.train_results['confusion_matrix'],
                                      dataset.label_values, vis=vis)

        # Load test model
        x_test, y_test, test_indices = build_dataset(dataset.img,
                                                     test_gt,
                                                     ignored_labels=dataset.ignored_labels)
        x_test_pca, _, _ = build_dataset(dataset.img_pca,
                                         test_gt,
                                         ignored_labels=dataset.ignored_labels)

        # Test model
        quantum_svm.test(
            test_data=x_test_pca,
            test_gt=y_test,
            train_data=x_pca,
            label_names=[j for i, j in enumerate(dataset.label_values)
                         if i not in dataset.ignored_labels], load_model=True)
        pretty_print_confusion_matrix(quantum_svm.test_results['confusion_matrix'],
                                      dataset.label_values, vis=vis)

        # Convert prediction to original image shape
        gt_train_pred = pred_to_gt(
            train_indices,
            quantum_svm.train_results['predictions'],
            test_gt.shape,
        )
        gt_test_pred = pred_to_gt(
            test_indices,
            quantum_svm.test_results['predictions'],
            test_gt.shape,
        )
        gt_pred = gt_train_pred + gt_test_pred

        # Display results with visdom
        display_train_test_split(
            dataset.gt,
            gt_train_pred,
            gt_test_pred,
            "Train Prediction",
            "Test Prediction",
            dataset.label_values,
            vis)

        display_train_test_split(
            dataset.gt,
            dataset.gt,
            gt_pred,
            "Ground Truth",
            "Prediction",
            dataset.label_values,
            vis)

    elif MODEL == 'edge_detection':
        pass

    elif MODEL == 'edge_detection_quantum':
        pass

    elif MODEL == 'quantum_SVM_filter':
        pass

    elif MODEL == 'QNN':
        pass

    elif MODEL == 'trainable_quantum_SVM':
        pass

    elif MODEL == 'QCNN':
        pass

    else:
        print('Please choose existing model')


if __name__ == "__main__":
    main()
