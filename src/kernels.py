# Docstring
import jax
import pennylane.numpy as np
import pennylane as qml
from tqdm import tqdm

KERNEL_CONFIG = {
    'linear': {
        'function': lambda x, y: np.dot(x, y),
        'gradient': lambda x, y: y,
        'hessian': lambda x, y: np.eye(x.shape[0])
    },
    'polynomial': {
        'function': lambda x, y: np.dot(x, y) ** 2,
        'gradient': lambda x, y: 2 * np.dot(x, y),
        'hessian': lambda x, y: 2 * np.eye(x.shape[0])
    },
    'rbf': {
        'function': lambda x, y: np.exp(-np.linalg.norm(x - y) ** 2),
        'gradient': lambda x, y: -2 * np.exp(-np.linalg.norm(x - y) ** 2) * (x - y),
        'hessian': lambda x, y: -2 * np.exp(-np.linalg.norm(x - y) ** 2) * np.eye(x.shape[0])
    },
    'sigmoid': {
        'function': lambda x, y: 1 / (1 + np.exp(-np.dot(x, y))),
        'gradient': lambda x, y: np.dot(x, y) * (1 - np.dot(x, y)),
        'hessian': lambda x, y: np.dot(x, y) * (1 - np.dot(x, y)) * np.eye(x.shape[0])
    },
    'quantum_kernel': {
    }


}

class QuantumKernel:
    def __init__(self, kernel_name: str, kernels=KERNEL_CONFIG) -> None:
        if kernel_name not in kernels.keys():
            raise ValueError("{} is not a valid kernel name.".format(kernel_name))

        self.kernel_name = kernel_name
        

    def get_quantumkernel(self):
        if self.kernel_name == 'quantum_kernel':
            return self.quantum_kernel

    # Adjoint kernel, only for pure states
    @staticmethod
    def quantum_kernel(feature_map, train_data, test_data=None):
        if test_data is None:
            test_data = train_data  # Training Gram matrix
        assert (
            train_data.shape[1] == test_data.shape[1]
        ), "The training and testing data must have the same dimensionality"
        N = train_data.shape[1]

        # create device using JAX
        device = qml.device("default.qubit.jax", wires=N)

        # create projector (measures probability of having all "00...0")
        projector = np.zeros((2**N, 2**N))
        projector[0, 0] = 1

        # alteratively, we can return the probabilities and pick the first one

        # define the circuit for the quantum kernel ("overlap test" circuit)
        @jax.jit
        @qml.qnode(device, interface='jax', seed=42)
        def kernel(x1, x2):
            feature_map(x1, wires=range(N))
            qml.adjoint(feature_map)(x2, wires=range(N))
            return qml.expval(qml.Hermitian(projector, wires=range(N)))

        # build the gram matrix
        gram = np.zeros(shape=(test_data.shape[0], train_data.shape[0]))
        if test_data.shape[0] == train_data.shape[0]:
            for i in tqdm(range(test_data.shape[0])):
                for j in range(i, train_data.shape[0]):
                    gram[i][j] = kernel(test_data[i], train_data[j])
                    gram[j][i] = gram[i][j] # Matrix is symmetric

        else:
            for i in tqdm(range(test_data.shape[0])):
                for j in range(train_data.shape[0]):
                    gram[i][j] = kernel(test_data[i], train_data[j])
                    # gram[j][i] = gram[i][j] # Matrix is symmetric

        #TODO: CERN people mixed up order of nested for loop, extended to different train/test data



        return gram