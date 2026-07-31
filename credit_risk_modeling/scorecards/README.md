# Credit Score Scaling

## Overview

A logistic regression model naturally produces probabilities or log-odds. 
In a credit scorecard, these outputs are transformed into an interpretable numerical score.

The purpose of `ScoreScaler` is to convert:

* Probability of Default (PD) → Score
* Log-Odds → Score
* Score → Probability of Default

It can also generate scorecard points associated with each WOE bin.

---

## Credit Scoring Framework

A scorecard is based on the relationship:

```text
Odds = P(Good) / P(Bad)
```

and

```text
LogOdds = ln(Odds)
```

The final score is computed as:

```text
Score = Offset + Factor × LogOdds
```

where:

```text
Factor = PDO / ln(2)
```

and

```text
Offset = BaseScore - Factor × ln(BaseOdds)
```

---

## Main Components

### Base Score

The reference score assigned to a customer having the reference odds.

Example:

```python
base_score = 600
```

A customer with the specified base odds will receive a score of 600.

---

### Base Odds

Reference odds at the base score.

Example:

```python
base_odds = 50
```

means:

```text
Odds = 50 : 1
```

or

```text
P(Good) / P(Bad) = 50
```

The corresponding probability of default is:

```text
PD = 1 / (1 + 50)
   ≈ 1.96%
```

---

### PDO (Points to Double the Odds)

PDO defines how many score points correspond to a doubling of the odds.

Example:

```python
pdo = 20
```

means:

| Odds      | Score Change |
| --------- | ------------ |
| 25 → 50   | +20          |
| 50 → 100  | +20          |
| 100 → 200 | +20          |

The larger the PDO, the flatter the score scale.

---

### Factor

The scaling factor is:

```text
Factor = PDO / ln(2)
```

For:

```python
pdo = 20
```

we obtain:

```text
Factor ≈ 28.85
```

---

### Offset

The offset ensures that the chosen base score corresponds exactly to the chosen base odds.

```text
Offset = BaseScore - Factor × ln(BaseOdds)
```

For:

```python
base_score = 600
base_odds = 50
pdo = 20
```

the offset is automatically computed.

---

## Probability to Score

The workflow is:

```text
PD
 ↓
Odds
 ↓
LogOdds
 ↓
Score
```

using:

```text
Odds = (1 - PD) / PD
```

```text
LogOdds = ln(Odds)
```

```text
Score = Offset + Factor × LogOdds
```

Higher scores correspond to lower probabilities of default.

---

## Score to Probability

The inverse transformation is:

```text
LogOdds = (Score - Offset) / Factor
```

```text
Odds = exp(LogOdds)
```

```text
PD = 1 / (1 + Odds)
```

This allows the original probability of default to be recovered from a score.

---

## Scorecard Points

When logistic regression is trained on WOE-transformed variables:

```text
LogOdds = Intercept + Σ(beta_j × WOE_j)
```

the contribution of each bin can be converted into score points.

For a given feature:

```text
Points_j = -Factor × beta_j × WOE_j
```

where:

* `beta_j` is the logistic regression coefficient
* `WOE_j` is the Weight of Evidence of the bin
* `Factor` is the score scaling factor

The final score becomes:

```text
Score = BasePoints + Σ(Points_j)
```

This decomposition makes the scorecard fully interpretable and auditable.

---

## Typical Credit Scoring Pipeline

```text
Raw Data
    ↓
Binning
    ↓
WOE Transformation
    ↓
Logistic Regression
    ↓
PD Estimation
    ↓
Score Scaling
    ↓
Credit Score
```

The `ScoreScaler` class implements the final stage of this pipeline.

---

## Implemented Methods

| Method                      | Description                                                                     |
| --------------------------- | ------------------------------------------------------------------------------- |
| `log_odds_to_score()`       | Converts log-odds into a credit score                                           |
| `probability_to_score()`    | Converts PD into a credit score                                                 |
| `score_to_probability()`    | Converts a score back into PD                                                   |
| `create_scorecard_points()` | Generates scorecard points from logistic regression coefficients and WOE values |

---

## References

* Siddiqi, N. *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*
* Anderson, R. *The Credit Scoring Toolkit*
* Basel Committee on Banking Supervision
* Scikit-learn Logistic Regression Documentation
