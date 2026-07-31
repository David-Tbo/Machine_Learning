#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =============================================
# 1. Utility functions for the decision tree
# =============================================

def split_tree(tree: pd.DataFrame, criteria: str, threshold: float) -> tuple:
    """Split a DataFrame into left and right subtrees based on a threshold."""
    left_tree = tree[tree[criteria] < threshold]
    right_tree = tree[tree[criteria] >= threshold]
    return left_tree, right_tree

def entropy_node(tree: pd.DataFrame, binary_target: str) -> float:
    """Calculate the entropy of a node (uncertainty)."""
    n = tree.shape[0]
    if n == 0:
        return 0
    p = tree[binary_target].mean()
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def entropy_split(left_tree: pd.DataFrame, right_tree: pd.DataFrame, binary_target: str) -> float:
    """Calculate the weighted entropy after a split."""
    n_left = left_tree.shape[0]
    n_right = right_tree.shape[0]
    n_total = n_left + n_right
    if n_total == 0:
        return 0
    weight_left = n_left / n_total
    weight_right = n_right / n_total
    weighted_entropy = (
        weight_left * entropy_node(left_tree, binary_target)
        + weight_right * entropy_node(right_tree, binary_target)
    )
    return weighted_entropy

def information_gain(tree: pd.DataFrame, criteria: str, threshold: float, binary_target: str) -> float:
    """Calculate the information gain from a split."""
    parent_entropy = entropy_node(tree, binary_target)
    left_tree, right_tree = split_tree(tree, criteria, threshold)
    split_entropy = entropy_split(left_tree, right_tree, binary_target)
    return parent_entropy - split_entropy

def find_best_threshold(tree: pd.DataFrame, criteria: str, binary_target: str) -> tuple:
    """Find the best threshold for a feature to maximize information gain."""
    thresholds = tree[criteria].unique()
    best_gain = -np.inf
    best_threshold = None
    for threshold in thresholds:
        gain = information_gain(tree, criteria, threshold, binary_target)
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    return best_threshold, best_gain

# =============================================
# 2. Decision Tree Construction
# =============================================

def majority_class(tree: pd.DataFrame, binary_target: str) -> int:
    """Return the majority class in a node."""
    return int(tree[binary_target].mean() >= 0.5)

def build_tree(
    tree: pd.DataFrame,
    features: list,
    binary_target: str,
    depth: int = 0,
    max_depth: int = 3,
    min_samples_split: int = 5
) -> dict:
    """Recursively build a decision tree."""
    if (
        entropy_node(tree, binary_target) == 0
        or depth >= max_depth
        or len(tree) < min_samples_split
    ):
        return {
            "type": "leaf",
            "prediction": majority_class(tree, binary_target)
        }

    best_feature = None
    best_threshold = None
    best_gain = -np.inf

    for feature in features:
        threshold, gain = find_best_threshold(tree, feature, binary_target)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            best_threshold = threshold

    if best_gain <= 0:
        return {
            "type": "leaf",
            "prediction": majority_class(tree, binary_target)
        }

    left_tree, right_tree = split_tree(tree, best_feature, best_threshold)

    return {
        "type": "node",
        "feature": best_feature,
        "threshold": best_threshold,
        "gain": best_gain,
        "left": build_tree(
            left_tree,
            features,
            binary_target,
            depth=depth + 1,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        ),
        "right": build_tree(
            right_tree,
            features,
            binary_target,
            depth=depth + 1,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
    }

# =============================================
# 3. Prediction with the Decision Tree
# =============================================

def predict_one(row: pd.Series, tree_model: dict) -> int:
    """Predict the class for a single row using the decision tree."""
    if tree_model["type"] == "leaf":
        return tree_model["prediction"]
    feature = tree_model["feature"]
    threshold = tree_model["threshold"]
    if row[feature] < threshold:
        return predict_one(row, tree_model["left"])
    else:
        return predict_one(row, tree_model["right"])

def predict(tree_model: dict, X_test: pd.DataFrame) -> np.ndarray:
    """Predict classes for a test dataset."""
    return X_test.apply(lambda row: predict_one(row, tree_model), axis=1)

# =============================================
# 4. Main Function
# =============================================

def main():
    # Load the Breast Cancer dataset from sklearn
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create a DataFrame for training with the target
    df_train = pd.concat([X_train, y_train], axis=1)
    features = [col for col in df_train.columns if col != "target"]

    # Build the decision tree
    tree_model = build_tree(
        tree=df_train,
        features=features,
        binary_target="target",
        max_depth=3,
        min_samples_split=10
    )

    # Predict on the test set
    y_pred = predict(tree_model, X_test)

    # Calculate and display accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Display a prediction example
    example_row = X_test.iloc[0]
    example_prediction = predict_one(example_row, tree_model)
    print(f"\nPrediction example for the first test row: {example_prediction}")
    print(f"Actual value: {y_test.iloc[0]}")

if __name__ == "__main__":
    main()