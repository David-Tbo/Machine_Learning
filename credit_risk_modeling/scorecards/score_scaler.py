"""
Scorecard scaling utilities.

This module converts logistic regression log-odds into credit scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ScoreScaler:
    """
    Credit score scaler.

    Parameters
    ----------
    base_score : int
        Reference score.

    base_odds : float
        Reference odds Good/Bad at base_score.

    pdo : float
        Points to double the odds.
    """

    def __init__(
        self,
        base_score: int = 600,
        base_odds: float = 50.0,
        pdo: float = 20.0,
    ) -> None:
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo

        self.factor_: float = self.pdo / np.log(2)
        self.offset_: float = self.base_score - self.factor_ * np.log(self.base_odds)

    def log_odds_to_score(self, log_odds_good_bad: np.ndarray) -> np.ndarray:
        """
        Convert log-odds Good/Bad to score.
        """
        return self.offset_ + self.factor_ * log_odds_good_bad

    def probability_to_score(self, pd_values: np.ndarray) -> np.ndarray:
        """
        Convert probability of default to score.

        PD is P(Bad). Odds are Good/Bad = (1-PD)/PD.
        """
        pd_values = np.clip(pd_values, 1e-8, 1 - 1e-8)
        odds_good_bad = (1 - pd_values) / pd_values
        log_odds = np.log(odds_good_bad)

        return self.log_odds_to_score(log_odds)

    def score_to_probability(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert score back to probability of default.
        """
        log_odds_good_bad = (scores - self.offset_) / self.factor_
        odds_good_bad = np.exp(log_odds_good_bad)

        return 1 / (1 + odds_good_bad)

    def create_scorecard_points(
        self,
        coefficients: pd.Series,
        woe_tables: dict[str, pd.DataFrame],
        ) -> pd.DataFrame:
        """
        Compute scorecard points by variable and bin.

        Logistic regression on WOE-transformed variables gives:

            log_odds = intercept + sum(beta_j * WOE_j)

        The score is defined as:

            score = Offset + Factor * log_odds

        with:

            Factor = PDO / ln(2)
            Offset = base_score - Factor * ln(base_odds)

        Therefore, the partial score assigned to each bin is:

            points_j = -Factor * beta_j * WOE_j

        Returns
        -------
        pd.DataFrame
            Table containing one row per feature bin with the WOE,
            coefficient and associated scorecard points.
            """
        
        rows = []

        for feature, table in woe_tables.items():
            if feature not in coefficients:
                continue

            beta = coefficients[feature]

            for bin_label, row in table.iterrows():
                partial_score = -self.factor_ * beta * row["woe"]

                rows.append(
                    {
                        "feature": feature,
                        "bin": str(bin_label),
                        "woe": row["woe"],
                        "coefficient": beta,
                        "points": partial_score,
                    }
                )

        return pd.DataFrame(rows)