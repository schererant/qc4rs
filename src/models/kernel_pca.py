import time
from tracemalloc import start
from numpy import short
import pennylane.numpy as np
import pennylane as qml
from sklearn.svm import SVC
import sklearn.decomposition as skd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import dill as pickle
import jax
import os

# Define Embedding layer

class Kernel:
    def __init__(self, num_wires, num_layers, device='default.qubit'):
        # self.X = X
        # self.Y = Y
        # self.kernel_type = kernel_type
        self.params = np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires), requires_grad=True)
        self.dev = qml.device(device, wires=num_wires, shots=None)
        self.wires = self.dev.wires.tolist()
        


    def feature_map(self, feature_0, feature_1, params):
        
        # Define layer/s
        def layer(feature, params, wires, i0=0, inc=1):
            i = i0
            for j, wire in enumerate(wires):
                qml.Hadamard(wires=wire)
                qml.RZ(feature[i % len(feature)], wires=wire)
                i += inc
                qml.RY(params[0, j], wires=wire)
            
            qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])

        # Define Ansatz
        def ansatz(feature, params, wires):
            for j, layer_params in enumerate(params):
                layer(feature, layer_params, wires, i0=j*len(wires))

        # Define adjoint
        adjoint_ansatz = qml.adjoint(ansatz)

        
        # Define circuit
        # print("params is None")
        ansatz(feature_0, params, self.wires)
        adjoint_ansatz(feature_1, params, self.wires)
        # else:
        #     # print("params is not None")
        #     ansatz(feature_0, params, self.wires)
        #     adjoint_ansatz(feature_1, params, self.wires)


    def kernel(self, feature_0, feature_1, params, print_kernel=False):
        # Create device
        # @jax.jit
        self.params = params
        dev = qml.device('lightning.qubit', wires=self.wires, shots=None)
        @qml.qnode(dev)
        def kernel_circuit(feature_0, feature_1, params):
            self.feature_map(feature_0, feature_1, params)
            return qml.probs(self.wires)

        # if print_circuit:
        # print(qml.drawer(kernel_circuit)(feature_0, feature_1))
        
        return kernel_circuit(feature_0, feature_1, params)[0]

    def get_lambda_kernel(self):
        # Define Kernel as lambda function
        lambda_kernel = lambda x1, x2: self.kernel(x1, x2, self.params)

        return lambda_kernel

    def get_square_kernel_matrix(self, features):
        # Define Kernel as lambda function
        lambda_kernel = lambda x1, x2: self.kernel(x1, x2, self.params)
        
        # Create Kernel Matrix
        K = qml.kernels.square_kernel_matrix(features, lambda_kernel)

        return K
        
    def print_square_kernel_matrix(self, features):
        # Define Kernel as lambda function
        lambda_kernel = lambda x1, x2: self.kernel(x1, x2, self.params)
        
        # Create Kernel Matrix
        K = qml.kernels.square_kernel_matrix(features, lambda_kernel)

        with np.printoptions(precision=3, suppress=True):
            print(K)


class Dataset:
    def __init__(self, truncate=10, binary=False):
        self.train_data, self.train_labels, self.test_data, self.test_labels = self.get_data(truncate=truncate, binary=binary)

    def get_data(self, dataset='paviaU', pca_components=2, standard_scalar=True, truncate=None, test_size=0.8, random_state=42, binary=False):
        """
        Import pavia university dataset as a numpy array
        """
        if dataset == 'paviaU':
            # Import pavia university dataset as a numpy array
            script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
            rel_path = "../../data"
            abs_file_path = os.path.join(script_dir, rel_path)
            features = loadmat(abs_file_path+'/PaviaU.mat')['paviaU']
            labels = loadmat(abs_file_path+'/PaviaU_gt.mat')['paviaU_gt']

        data = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        labels = np.reshape(labels, (labels.shape[0] * labels.shape[1],))

        if binary:
            labels = np.array([-1 if x == 0 else 1 for x in labels])

        
        if pca_components > 0:
            # Perform PCA on the data
            pca = skd.PCA(n_components=pca_components)
            pca.fit(data)
            data = pca.transform(data)

        if standard_scalar:
            data = StandardScaler().fit_transform(data)

        

        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size, random_state=random_state)

        if truncate is not None:
            train_data = train_data[:truncate]
            train_labels = train_labels[:truncate]
            test_data = test_data[:truncate]
            test_labels = test_labels[:truncate]

        
            # for i in range(len(train_labels)):
            #     if train_labels[i] > 0:
            #         train_labels[i] = 1
            #     else:
            #         train_labels[i] = -1

        return train_data, train_labels, test_data, test_labels

class Classifier:
    def __init__(self, kernel, dataset):
        self.model = None
        self.predictions = None
        self.kernel = kernel
        self.init_kernel = kernel.get_lambda_kernel()
        self.dataset = dataset

    def train(self, report=False, load_model=False, save_model=True):
        # Kernel = Kernel(num_wires=features.shape[1], num_layers=num_layers, device=device)
        train_data = {'samples': [], 'train': [], 'predict': [], 'accuracy': []}
        train_data['samples'] = len(self.dataset.train_data)

        if not load_model:
            start = time.time()
            self.model = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, self.init_kernel))
            self.model.fit(self.dataset.train_data , self.dataset.train_labels)
            end = time.time()
            train_data['train'].append(end - start)
            print(f"SVM trained in {end - start:.3f} seconds")
            if save_model:
                pickle.dump(self.model, open('svm.pkl', 'wb'))
        else:
            self.model = pickle.load(open('svm.pkl', 'rb'))
            print('Model loaded')
        
        # Inference
        start = time.time()
        self.predictions = self.model.predict(self.dataset.train_data)
        end = time.time()
        train_data['predict'].append(end - start)
        print(f"SVM inference in {end - start:.3f} seconds")

        accuracy_init = 1 - np.count_nonzero(self.predictions - self.dataset.train_labels) / len(self.dataset.train_labels)
        print(f"The accuracy of the kernel with random parameters is {accuracy_init:.3f}")
        #     print(f"SVM accuracy time: {end - start:.3f} seconds")
        train_data['accuracy'].append(accuracy_init)
        
        if report:
            print('Accuracy:', self.model.score(self.dataset.train_data, self.dataset.train_labels))
            print('Confusion Matrix:')
            print(confusion_matrix(self.dataset.train_labels, self.model.predict(self.dataset.train_data)))
            print('Classification Report:')
            print(classification_report(self.dataset.train_labels, self.model.predict(self.dataset.train_data), zero_division=0))

        return train_data

    def plot_predicted_points(self):
        plt.scatter(self.dataset.train_data[:, 0], self.dataset.train_data[:, 1], c=self.predictions, cmap='rainbow')
        plt.show()

    def evaluate_kernel_target_alignment(self):
        start = time.time()
        kernel_target_alignment = qml.kernels.target_alignment(self.dataset.train_data, self.dataset.train_labels, self.init_kernel)
        print(f"The target alignment of the kernel with random parameters is {kernel_target_alignment:.3f}")
        end = time.time()
        print(f"Target alignment of the kernel with random parameters in {end - start:.3f} seconds")

    def train_target_alignment(self):
        def target_alignment(
            X,
            Y,
            kernel,
            assume_normalized_kernel=False,
            rescale_class_labels=True,
            ):
            """Kernel-target alignment between kernel and labels."""

            K = qml.kernels.square_kernel_matrix(
                X,
                kernel,
                assume_normalized_kernel=assume_normalized_kernel,
            )

            if rescale_class_labels:
                nplus = np.count_nonzero(np.array(Y) == 1)
                nminus = len(Y) - nplus
                _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
            else:
                _Y = np.array(Y)

            T = np.outer(_Y, _Y)
            inner_product = np.sum(K * T)
            norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
            inner_product = inner_product / norm

            return inner_product

        params = self.kernel.params
        # print('Initial kernel parameters:', params)
        # opt = qml.GradientDescentOptimizer(0.5)
        opt = qml.AdamOptimizer(0.01)

        for i in range(50):
            # Choose subset of datapoints to compute the KTA on.
            start = time.time()
            subset = np.random.choice(list(range(len(self.dataset.train_data))), 4)
            end = time.time()
            print(f"Subset selection in {end - start:.3f} seconds")
            # Define the cost function for optimization
            start = time.time()
            cost = lambda _params: -target_alignment(
                self.dataset.train_data[subset],
                self.dataset.train_labels[subset],
                lambda x1, x2: self.kernel.kernel(x1, x2, _params),
                assume_normalized_kernel=True,
            )
            end = time.time()
            print(f"Cost function definition in {end - start:.3f} seconds")
            # Optimization step
            start = time.time()
            print(params.shape)
            params = opt.step(cost, params)
            end = time.time()
            print(f"Optimization step in {end - start:.3f} seconds")
            

            # Report the alignment on the full dataset every 50 steps.
            if (i + 1) % 5 == 0:
                current_alignment = target_alignment(
                    self.dataset.train_data,
                    self.dataset.train_labels,
                    lambda x1, x2: self.kernel.kernel(x1, x2, params),
                    assume_normalized_kernel=True,
                )
                print(f"Step {i+1} - Alignment = {current_alignment:.3f}")



# dataset = Dataset(truncate=5, binary=True)
# print(dataset.train_labels)
# kernel = Kernel(num_layers=6, num_wires=5)
# classifier = Classifier(kernel, dataset)
# classifier.train(report=False)
# # train(features=train_data, labels=train_labels, Kernel=kernel, num_layers=6, report=True)
# classifier.evaluate_kernel_target_alignment()
# classifier.train_target_alignment()
