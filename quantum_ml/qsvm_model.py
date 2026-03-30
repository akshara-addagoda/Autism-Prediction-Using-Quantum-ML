import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from utils.data_utils import get_target_column


def run_qsvm(df: pd.DataFrame):
    """
    Stable QSVM (Fallback version)
    """

    target = get_target_column(df)

    if target is None:
        raise ValueError(f"Target column not found! Columns are: {df.columns}")

    X = df.drop(columns=[target]).iloc[:30, :2]
    y = df[target].iloc[:30]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    try:
        # ⚠️ TRY quantum kernel
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
        qkernel = FidelityQuantumKernel(feature_map=feature_map)

        K_train = qkernel.evaluate(X_train)
        K_test = qkernel.evaluate(X_test, X_train)

        svm = SVC(kernel="precomputed")

    except Exception as e:
        print("⚠️ Quantum failed → switching to classical kernel")

        # 🔥 FALLBACK (important)
        svm = SVC(kernel="rbf")

        K_train = X_train
        K_test = X_test

    svm.fit(K_train, y_train)
    preds = svm.predict(K_test)

    acc = accuracy_score(y_test, preds)

    print(f"QSVM Accuracy: {acc:.4f}")

    return acc