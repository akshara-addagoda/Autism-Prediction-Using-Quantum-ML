import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from utils.data_loader import load_data
from utils.data_utils import get_target_column


def run_svm(user_input):
    """
    Support Vector Machine (SVM)
    Returns: prediction, accuracy, confidence
    """

    # ✅ Load dataset
    df = load_data()

    # ✅ Detect target column safely
    target_col = get_target_column(df)
    if target_col is None:
        raise ValueError("Target column not found!")

    # ✅ Select first 10 features (Q-CHAT)
    X = df.drop(columns=[target_col]).iloc[:, :10]
    y = df[target_col]

    # ✅ Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # ✅ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 🔥 IMPORTANT: reshape + scale user input
    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)

    # ✅ Train SVM
    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)

    # ✅ Prediction
    pred_encoded = model.predict(user_input)[0]
    pred = le.inverse_transform([pred_encoded])[0]  # 🔥 FIXED label output

    # ✅ Confidence
    conf = float(np.max(model.predict_proba(user_input)) * 100)

    # ✅ Accuracy
    acc = accuracy_score(y_test, model.predict(X_test))

    return pred, acc, conf