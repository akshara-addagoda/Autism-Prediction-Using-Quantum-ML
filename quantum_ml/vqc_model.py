import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# ✅ Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import COBYLA

# ✅ IMPORT THIS (VERY IMPORTANT)
from utils.data_utils import get_target_column


def run_vqc(df: pd.DataFrame):
    """
    Variational Quantum Classifier (VQC)
    """

    # ✅ FIXED target detection
    target = get_target_column(df)

    if target is None:
        raise ValueError(f"Target column not found! Columns are: {df.columns}")

    # ✅ Reduce dataset
    X = df.drop(columns=[target]).iloc[:40, :2]
    y = df[target].iloc[:40]

    # ✅ Encode labels (IMPORTANT FIX)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # ✅ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ✅ Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ✅ Quantum circuit
    n_qubits = 2
    inputs = ParameterVector("x", n_qubits)
    weights = ParameterVector("θ", n_qubits)

    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.h(i)
        qc.ry(inputs[i], i)
        qc.ry(weights[i], i)

    # ✅ Backend
    estimator = Estimator()

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        estimator=estimator
    )

    # ✅ Classifier
    clf = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=COBYLA(maxiter=10)
    )

    # ✅ Train
    clf.fit(X_train, y_train)

    # ✅ Predict
    preds = clf.predict(X_test)

    # ✅ Accuracy
    acc = accuracy_score(y_test, preds)

    print(f"VQC Accuracy: {acc:.4f}")

    return acc