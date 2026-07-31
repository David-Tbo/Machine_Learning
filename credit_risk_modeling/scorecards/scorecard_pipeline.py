"""
Industrial scorecard model.

This module combines:
- numerical monotonic WoE binning
- categorical WoE encoding
- logistic regression
- score scaling
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from monotonic_binner import MonotonicBinner
from woe_encoder import WoEEncoder
from score_scaler import ScoreScaler


class ScorecardModel(BaseEstimator, ClassifierMixin):
    """
    Production-oriented scorecard model.

    Parameters
    ----------
    numerical_binner : MonotonicBinner
        Numerical binning and WoE encoder.

    categorical_encoder : WoEEncoder
        Categorical WoE encoder.

    logistic_model : LogisticRegression
        Logistic regression model.

    score_scaler : ScoreScaler
        Score scaling object.
    """

    def __init__(
        self,
        numerical_binner: MonotonicBinner | None = None,
        categorical_encoder: WoEEncoder | None = None,
        logistic_model: LogisticRegression | None = None,
        score_scaler: ScoreScaler | None = None,
    ) -> None:
        self.numerical_binner = numerical_binner or MonotonicBinner()
        self.categorical_encoder = categorical_encoder or WoEEncoder()
        self.logistic_model = logistic_model or LogisticRegression(max_iter=1000)
        self.score_scaler = score_scaler or ScoreScaler()

        self.numerical_features_: list[str] = []
        self.categorical_features_: list[str] = []
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ScorecardModel":
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).astype(int)

        self.numerical_features_ = X.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(exclude=["number"]).columns.tolist()

        X_num = X[self.numerical_features_]
        X_cat = X[self.categorical_features_]

        X_num_woe = self.numerical_binner.fit_transform(X_num, y)

        if len(self.categorical_features_) > 0:
            X_cat_woe = self.categorical_encoder.fit_transform(X_cat, y)
            X_woe = pd.concat([X_num_woe, X_cat_woe], axis=1)
        else:
            X_woe = X_num_woe

        self.feature_names_ = X_woe.columns.tolist()

        self.logistic_model.fit(X_woe, y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()

        X_num = X[self.numerical_features_]
        X_num_woe = self.numerical_binner.transform(X_num)

        if len(self.categorical_features_) > 0:
            X_cat = X[self.categorical_features_]
            X_cat_woe = self.categorical_encoder.transform(X_cat)
            X_woe = pd.concat([X_num_woe, X_cat_woe], axis=1)
        else:
            X_woe = X_num_woe

        return X_woe[self.feature_names_]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_woe = self.transform(X)
        return self.logistic_model.predict_proba(X_woe)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_woe = self.transform(X)
        return self.logistic_model.predict(X_woe)

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        pd_values = self.predict_proba(X)[:, 1]
        return self.score_scaler.probability_to_score(pd_values)

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.feature_names_,
                "coefficient": self.logistic_model.coef_[0],
            }
        )

    def auc(self, X: pd.DataFrame, y: pd.Series) -> float:
        y = pd.Series(y).astype(int)
        pred = self.predict_proba(X)[:, 1]
        return float(roc_auc_score(y, pred))