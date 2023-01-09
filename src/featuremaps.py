import pennylane as qml
import pennylane.numpy as np


FEATUREMAPS_CONFIG = {
    'zz_entangled':{

    }
}

class FeatureMap:

    def __init__(self, featuremap_name: str, featuremaps=FEATUREMAPS_CONFIG) -> callable:

        self.featuremap_name = featuremap_name
        self.featuremaps = featuremaps

    def get_feature_map(self):
        if self.featuremap_name not in self.featuremaps.keys():
            raise ValueError("{} is not a valid featuremap name.".format(self.featuremap_name))

        if self.featuremap_name == 'zz_entangled':
            return self.zz_entangled


    ################################################################################
    # Static
    ################################################################################
    @staticmethod
    def zz_entangled(x, wires):
        N = len(wires)
        for i in range(N):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[i], wires=i)
        for i in range(N):
            for j in range(i + 1, N):
                qml.CRZ(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=[i, j])


    ################################################################################
    # Paramterized
    ################################################################################
    @staticmethod
    def hardware_efficient_ansatz(theta, wires):
        N = len(wires)
        assert len(theta) == 2 * N
        for i in range(N):
            qml.RX(theta[2 * i], wires=wires[i])
            qml.RY(theta[2 * i + 1], wires=wires[i])
        for i in range(N - 1):
            qml.CZ(wires=[wires[i], wires[i + 1]])

    ################################################################################
    # Continous
    ################################################################################
