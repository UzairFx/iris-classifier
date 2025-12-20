import argparse
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def main(test_size: float = 0.2, random_state: int = 42) -> None:
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train model
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Predict + accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Ensure outputs folder exists
    os.makedirs("outputs", exist_ok=True)

    # Save confusion matrix PNG
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot()
    plt.title("Iris Decision Tree - Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: outputs/confusion_matrix.png")

    # Save trained model
    joblib.dump(model, "outputs/model.joblib")
    print("Saved: outputs/model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(test_size=args.test_size, random_state=args.random_state)


