# **Explication des Valeurs de Shapley en Interprétabilité de Modèles**

---

## **0. Rappels Mathématiques Utiles**
Pour bien comprendre les valeurs de Shapley, il est utile de rappeler quelques concepts de base en combinatoire et probabilités :

---
### **Combinaisons**
En combinatoire, le symbole **$C(n, k)$** ou **$\binom{n}{k}$** représente le nombre de **combinaisons** de $k$ éléments parmi $n $ éléments, **sans tenir compte de l'ordre**.
La formule est :
$$C(n, k) = \binom{n}{k} = \frac{n!}{k!(n-k)!}$$

---
### **Permutations**
Quand **l'ordre compte**, on parle de **permutations**. Le nombre de permutations de $k$ éléments parmi $ n $ est donné par :
$$P(n, k) = \frac{n!}{(n-k)!}$$

---
### **Nombre de sous-ensembles (parties)**
Le nombre de **parties** (sous-ensembles) dans un ensemble à $ n $ éléments est :
$$2^n$$
*Exemple* : Pour un ensemble de 3 *features* $\{x_1, x_2, x_3\}$, il existe $2^3 = 8$ sous-ensembles possibles (y compris l'ensemble vide).

---
### **Factorielle**
La factorielle d'un entier $n$, notée $n!$, est le produit de tous les entiers positifs inférieurs ou égaux à $n$ :
$$n! = n \times (n-1) \times \dots \times 1$$

---

---

## **1. Introduction aux Valeurs de Shapley**
Les **valeurs de Shapley** sont une méthode d'interprétabilité des modèles de *machine learning* basée sur la **théorie des jeux coopératifs**. Elles permettent de **quantifier la contribution de chaque *feature*** (caractéristique) à la prédiction d'un modèle, en tenant compte de **toutes les interactions possibles** entre les *features*.

### **Analogie avec la Théorie des Jeux**
- Chaque *feature* est un **joueur** dans un jeu coopératif.
- La **valeur de la coalition** $v(S)$ représente la prédiction du modèle pour un sous-ensemble $S$ de *features*.
- La **contribution marginale** d'une *feature* $i$ est la différence entre $v(S \cup \{i\})$ et $v(S)$.

### **Formule Générale**
La valeur de Shapley pour une *feature* $i$ est définie comme la **moyenne pondérée des contributions marginales** sur toutes les permutations possibles des *features* :

**En LaTeX :**  
$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \times (|N| - |S| - 1)!}{|N|!} (v(S \cup \{i\}) - v(S))$$


**En anglais :**
- $\phi_i(v)$: The Shapley value for player $i$ in the game with characteristic function $v$.
- $S$: A subset of the set of players $N$ excluding player $i$.
- $N \setminus \{i\}$: The set of all players except player $i$.
- $|S|$: The number of players in subset $S$.
- $|N|$: The total number of players.
- $|S|!$: The factorial of the number of players in subset $S$.
- $(|N| - |S| - 1)!$: The factorial of the number of players not in subset $S$ and excluding player $i$.
- $|N|!$: The factorial of the total number of players.
- $v(S \cup \{i\})$: The value of the characteristic function $v$ for the coalition $S$ including player $i$.
- $v(S)$: The value of the characteristic function $v$ for the coalition $S$ excluding player $i$.

**Où :**
- $N$ : Ensemble de toutes les *features*.
- $S$ : Sous-ensemble de $N$ **n'incluant pas** $i$.
- $|S|$ : Taille du sous-ensemble $S$.
- $|N|$ : Taille totale de l'ensemble $N$.
- $v(S)$ : Valeur de la fonction de prédiction pour le sous-ensemble $S$.
- $v(S \cup \{i\}) - v(S)$ : Contribution marginale de la *feature* $i$ lorsqu'elle est ajoutée à $S$.

---
### **Propriétés des Valeurs de Shapley**
Les valeurs de Shapley satisfont plusieurs propriétés importantes :
1. **Efficacité** : La somme des valeurs de Shapley de tous les joueurs est égale à la valeur de la grande coalition.
2. **Symétrie** : Deux joueurs qui contribuent de manière identique à toutes les coalitions ont la même valeur de Shapley.
3. **Nullité** : Un joueur qui n'apporte aucune contribution marginale a une valeur de Shapley nulle.
4. **Additivité** : Si deux jeux sont combinés, la valeur de Shapley d'un joueur est la somme de ses valeurs dans chaque jeu.

---
### **Exemple Simple de Calcul des Poids**
Prenons un jeu coopératif avec **3 joueurs (A, B, C)** et une fonction caractéristique $ v $ définie comme suit :

**Valeurs des coalitions**


Valeurs des coalitions


| Coalition       | Valeur $ v(S) $ |
|-----------------|-------------------|
| $ \emptyset $ | 0                 |
| $ \{A\} $     | 2                 |
| $ \{B\} $     | 3                 |
| $ \{C\} $     | 4                 |
| $ \{A, B\} $  | 6                 |
| $ \{A, C\} $  | 7                 |
| $ \{B, C\} $  | 8                 |
| $ \{A, B, C\} $| 10                |

#### **Calcul des Contributions Marginales**
Pour chaque permutation des joueurs, on calcule la contribution marginale de chaque joueur :

1. **Permutation A, B, C** :
   - A : $v(\{A\}) - v(\emptyset) = 2 - 0 = 2$
   - B : $v(\{A, B\}) - v(\{A\}) = 6 - 2 = 4$
   - C : $v(\{A, B, C\}) - v(\{A, B\}) = 10 - 6 = 4$

2. **Permutation A, C, B** :
   - A : $2$
   - C : $v(\{A, C\}) - v(\{A\}) = 7 - 2 = 5$
   - B : $v(\{A, B, C\}) - v(\{A, C\}) = 10 - 7 = 3$

3. **Permutation B, A, C** :
   - B : $3$
   - A : $v(\{A, B\}) - v(\{B\}) = 6 - 3 = 3$
   - C : $4$

4. **Permutation B, C, A** :
   - B : $3$
   - C : $v(\{B, C\}) - v(\{B\}) = 8 - 3 = 5$
   - A : $v(\{A, B, C\}) - v(\{B, C\}) = 10 - 8 = 2$

5. **Permutation C, A, B** :
   - C : $4$
   - A : $v(\{A, C\}) - v(\{C\}) = 7 - 4 = 3$
   - B : $3$

6. **Permutation C, B, A** :
   - C : $4$
   - B : $ v(\{B, C\}) - v(\{C\}) = 8 - 4 = 4$
   - A : $2$

#### **Valeurs de Shapley**
Les valeurs de Shapley sont la **moyenne des contributions marginales** sur toutes les permutations :
- $\phi_A = \frac{2 + 2 + 3 + 2 + 3 + 2}{6} = \frac{14}{6} = \frac{7}{3}$
- $\phi_B = \frac{4 + 3 + 3 + 3 + 3 + 4}{6} = \frac{20}{6} = \frac{10}{3}$
- $\phi_C = \frac{4 + 5 + 4 + 5 + 4 + 4}{6} = \frac{26}{6} = \frac{13}{3}$

---
---
## **2. Méthodes Classiques d'Estimation des Valeurs de Shapley**
Trois méthodes classiques utilisent les équations de la théorie des jeux coopératifs pour calculer les explications des prédictions de modèles :

---
### **Shapley Regression Values**
- **Description** : Ces valeurs mesurent l'importance des *features* pour les modèles linéaires en présence de multicolinéarité.
- **Méthode** : Nécessite de réentraîner le modèle sur tous les sous-ensembles de *features* $S \subseteq F$, où $F$ est l'ensemble de toutes les *features*.
- **Calcul** : Les valeurs de Shapley sont calculées comme une moyenne pondérée des différences de prédictions entre les modèles avec et sans chaque *feature*.
- **Propriété** : Méthode additive d'attribution des *features*.

---
### **Shapley Sampling Values**
- **Description** : Explique n'importe quel modèle en utilisant des approximations d'échantillonnage.
- **Méthode** :
  1. Applique des approximations d'échantillonnage à la formule des valeurs de Shapley.
  2. Approximation de l'effet de la suppression d'une variable en intégrant sur des échantillons du jeu de données d'entraînement.
- **Avantage** : Évite de réentraîner le modèle et permet de calculer moins de $2^{|F|}$ différences.
- **Propriété** : Méthode additive d'attribution des *features*.

---
### **Quantitative Input Influence**
- **Description** : Cadre plus large qui traite de plus que les attributions de *features*.
- **Méthode** : Propose une approximation d'échantillonnage des valeurs de Shapley, similaire aux *Shapley Sampling Values*.
- **Propriété** : Méthode additive d'attribution des *features*.

---
---
## **3. Exemple avec un Modèle de Régression Linéaire**

### **Dataset**
Un dataset simple avec **3 *features*** et **6 observations** :


Dataset de Régression Linéaire


| **Observation** | **Feature 1 (x₁)** | **Feature 2 (x₂)** | **Feature 3 (x₃)** | **Target (y)** |
|----------------|-------------------|-------------------|-------------------|---------------|
| 1              | 1                 | 2                 | 3                 | 6             |
| 2              | 2                 | 3                 | 4                 | 9             |
| 3              | 3                 | 4                 | 5                 | 12            |
| 4              | 4                 | 5                 | 6                 | 15            |
| 5              | 5                 | 6                 | 7                 | 18            |
| 6              | 6                 | 7                 | 8                 | 21            |

### **Modèle de Régression Linéaire**
Le modèle ajusté est :
$$y = x_1 + x_2 + x_3$$
**Coefficients :**
- $\beta_0 = 0$
- $\beta_1 = 1$
- $\beta_2 = 1$
- $\beta_3 = 1$

---
### **Calcul des Valeurs de Shapley pour l'Observation 1**
Pour l'observation **1** ($x_1 = 1 $, $ x_2 = 2 $, $ x_3 = 3$), les contributions marginales sont calculées comme suit :

#### **Feature 1 (x₁)**
Contribution marginale moyenne :
$$\phi_{x_1} = \frac{1}{3} \left( 1 + 1 + 1 \right) = 1$$

#### **Feature 2 (x₂)**
Contribution marginale moyenne :
$$\phi_{x_2} = \frac{1}{3} \left( 2 + 2 + 2 \right) = 2$$

#### **Feature 3 (x₃)**
Contribution marginale moyenne :
$$\phi_{x_3} = \frac{1}{3} \left( 3 + 3 + 3 \right) = 3$$

---
### **Résultat**
Les valeurs de Shapley pour l'observation 1 sont :
- **Feature 1 (x₁)** : **1**
- **Feature 2 (x₂)** : **2**
- **Feature 3 (x₃)** : **3**

---
---
## **4. Exemple avec un Modèle de Classification**

### **Dataset**
Un dataset avec **4 *features*** et **10 observations** :


Dataset de Classification


| **Observation** | **Feature 1 (x₁)** | **Feature 2 (x₂)** | **Feature 3 (x₃)** | **Feature 4 (x₄)** | **Target (y)** |
|----------------|-------------------|-------------------|-------------------|-------------------|---------------|
| 1              | 1                 | 2                 | 3                 | 4                 | 1             |
| 2              | 2                 | 3                 | 4                 | 5                 | 0             |
| 3              | 3                 | 4                 | 5                 | 6                 | 1             |
| 4              | 4                 | 5                 | 6                 | 7                 | 0             |
| 5              | 5                 | 6                 | 7                 | 8                 | 1             |
| 6              | 6                 | 7                 | 8                 | 9                 | 0             |
| 7              | 7                 | 8                 | 9                 | 10                | 1             |
| 8              | 8                 | 9                 | 10                | 11                | 0             |
| 9              | 9                 | 10                | 11                | 12                | 1             |
| 10             | 10                | 11                | 12                | 13                | 0             |

---
### **Modèle de Classification (Régression Logistique)**
Le modèle est défini par :
$$\text{logit}(y) = -10 + x_1 + x_2 + x_3 + x_4$$
**Coefficients :**
- $\beta_0 = -10$
- $\beta_1 = 1$
- $\beta_2 = 1$
- $\beta_3 = 1$
- $\beta_4 = 1$

---
### **Calcul des Valeurs de Shapley pour l'Observation 1**
Pour l'observation **1** ($x_1 = 1 $, $ x_2 = 2 $, $ x_3 = 3 $, $ x_4 = 4$), les contributions marginales sont calculées pour chaque *feature* en considérant tous les sous-ensembles possibles.

#### **Feature 1 (x₁)**
Contribution marginale moyenne :
$$\phi_{x_1} = \frac{1}{4} \left( \text{logit}(-10 + 1) - \text{logit}(-10) + \text{logit}(-10 + 1 + 2) - \text{logit}(-10 + 2) + \text{logit}(-10 + 1 + 3) - \text{logit}(-10 + 3) + \text{logit}(-10 + 1 + 4) - \text{logit}(-10 + 4) \right)$$

#### **Feature 2 (x₂)**
Contribution marginale moyenne :
$$\phi_{x_2} = \frac{1}{4} \left( \text{logit}(-10 + 2) - \text{logit}(-10) + \text{logit}(-10 + 1 + 2) - \text{logit}(-10 + 1) + \text{logit}(-10 + 2 + 3) - \text{logit}(-10 + 3) + \text{logit}(-10 + 2 + 4) - \text{logit}(-10 + 4) \right)$$

#### **Feature 3 (x₃)**
Contribution marginale moyenne :
$$\phi_{x_3} = \frac{1}{4} \left( \text{logit}(-10 + 3) - \text{logit}(-10) + \text{logit}(-10 + 1 + 3) - \text{logit}(-10 + 1) + \text{logit}(-10 + 2 + 3) - \text{logit}(-10 + 2) + \text{logit}(-10 + 3 + 4) - \text{logit}(-10 + 4) \right)$$

#### **Feature 4 (x₄)**
Contribution marginale moyenne :
$$\phi_{x_4} = \frac{1}{4} \left( \text{logit}(-10 + 4) - \text{logit}(-10) + \text{logit}(-10 + 1 + 4) - \text{logit}(-10 + 1) + \text{logit}(-10 + 2 + 4) - \text{logit}(-10 + 2) + \text{logit}(-10 + 3 + 4) - \text{logit}(-10 + 3) \right)$$

---
### **Résultat**
Les valeurs de Shapley pour l'observation 1 sont :
- **Feature 1 (x₁)** : $\phi_{x_1}$
- **Feature 2 (x₂)** : $\phi_{x_2}$
- **Feature 3 (x₃)** : $\phi_{x_3}$
- **Feature 4 (x₄)** : $\phi_{x_4}$

---
---
## **5. Types d'Explainers SHAP en Python**

Les valeurs SHAP peuvent être calculées en utilisant différents types d'explainers, selon le type de modèle utilisé. Voici les principaux explainers disponibles dans la bibliothèque SHAP :

---
### **Kernel SHAP**
- **Nom sous Python** : `KernelExplainer`
- **Description** : Utilise une approche basée sur le noyau pour estimer les valeurs SHAP. Fonctionne avec n'importe quel modèle, mais peut être lent pour les grands ensembles de données.
- **Utilisation** :
  ```python
  explainer = shap.KernelExplainer(model.predict, X_train)
  shap_values = explainer.shap_values(X_test)
  ```

---
### **Tree SHAP**
- **Nom sous Python** : `TreeExplainer`
- **Description** : Spécifiquement conçu pour les modèles basés sur les arbres (comme les forêts aléatoires et les arbres de décision). Très efficace et rapide.
- **Utilisation** :
  ```python
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(X_test)
  ```

---
### **Deep SHAP**
- **Nom sous Python** : `DeepExplainer`
- **Description** : Conçu pour les réseaux de neurones profonds.
- **Utilisation** :
  ```python
  explainer = shap.DeepExplainer(model, X_train)
  shap_values = explainer.shap_values(X_test)
  ```

---
### **Linear SHAP**
- **Nom sous Python** : `LinearExplainer`
- **Description** : Utilisé pour les modèles linéaires. Simple et rapide.
- **Utilisation** :
  ```python
  explainer = shap.LinearExplainer(model, X_train)
  shap_values = explainer.shap_values(X_test)
  ```

---
### **Gradient SHAP**
- **Nom sous Python** : `GradientExplainer`
- **Description** : Utilise les gradients pour estimer les valeurs SHAP. Particulièrement utile pour les réseaux de neurones avec des fonctions d'activation différentiables.
- **Utilisation** :
  ```python
  explainer = shap.GradientExplainer(model, X_train)
  shap_values = explainer.shap_values(X_test)
  ```

---
### **Explainer Générique**
- **Nom sous Python** : `Explainer`
- **Description** : Interface unifiée qui détecte automatiquement le type de modèle et utilise l'explainer approprié.
- **Utilisation** :
  ```python
  explainer = shap.Explainer(model, X_train)
  shap_values = explainer(X_test)
  ```

---
---
## **6. Visualisation des Valeurs SHAP**

Les valeurs SHAP peuvent être visualisées de différentes manières pour mieux comprendre leur impact sur les prédictions du modèle.

---
### **Force Plot**
- **Description** : Montre comment chaque *feature* contribue à la prédiction d'une instance spécifique.
- **Utilisation** :
  ```python
  shap.initjs()
  shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
  ```

---
### **Summary Plot**
- **Description** : Montre l'importance globale des *features* et leur impact sur les prédictions.
- **Utilisation** :
  ```python
  shap.summary_plot(shap_values, X_test)
  ```

---
### **Dependency Plot**
- **Description** : Montre la relation entre la valeur d'une *feature* et son impact sur la prédiction, tout en colorant les points en fonction d'une autre *feature*.
- **Utilisation** :
  ```python
  shap.dependence_plot("mean radius", shap_values, X_test)
  ```

---
---
## **7. Détection de Biais avec les Valeurs SHAP**

Les valeurs SHAP peuvent être utilisées pour détecter des biais dans les modèles de *machine learning*. Voici quelques approches :

---
### **Importance Disproportionnée des *Features***
- **Description** : Si une *feature* a une importance disproportionnée par rapport à sa pertinence réelle, cela peut indiquer un biais.
- **Exemple** : Si `mean radius` a une importance très élevée par rapport à d'autres *features*, cela peut signifier que le modèle se repose trop sur cette *feature* pour faire des prédictions.

---
### **Relations Non Linéaires ou Seuils Abrupts**
- **Description** : Si les valeurs SHAP montrent des relations non linéaires ou des seuils abrupts pour certaines *features*, cela peut indiquer un biais.
- **Exemple** : Si `mean perimeter` a une relation non linéaire avec les valeurs SHAP, cela peut signifier que le modèle réagit de manière disproportionnée à certaines valeurs de cette *feature*.

---
### **Différences entre Sous-Groupes**
- **Description** : Si les valeurs SHAP montrent des différences significatives entre différents sous-groupes (par exemple, par âge, par sexe), cela peut indiquer un biais.
- **Exemple** : Si les valeurs SHAP pour `mean radius` sont très différentes pour les jeunes et les vieux, cela peut signifier que le modèle fait des prédictions différentes pour ces groupes.

---
### **Exemple de Code pour la Détection de Biais**
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

# Charger le dataset Breast Cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle de forêt aléatoire
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Créer un explainer SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Visualiser l'importance globale des *features*
shap.summary_plot(shap_values, X_test)

# Visualiser les dépendances des *features*
shap.dependence_plot("mean radius", shap_values, X_test)
```

---
---
## **8. Performance des Valeurs de Shapley**

L'utilisation des valeurs de Shapley pour expliquer les prédictions d'un modèle complexe est une méthode puissante pour rendre les modèles interprétables. Cependant, évaluer la performance des valeurs de Shapley dans ce contexte peut être plus subtil que d'évaluer la performance d'un modèle de *machine learning* traditionnel.

---
### **Consistance avec les Attentes**
Vérifiez si les valeurs de Shapley correspondent à vos attentes basées sur votre connaissance du domaine. Par exemple, si vous savez qu'une certaine *feature* devrait avoir un impact important sur la prédiction, les valeurs de Shapley devraient refléter cela.

---
### **Comparaison avec d'Autres Méthodes d'Explication**
Comparez les valeurs de Shapley avec d'autres méthodes d'explication de modèles, comme LIME (*Local Interpretable Model-agnostic Explanations*) ou les poids des *features* dans des modèles plus simples. Si les explications sont cohérentes entre différentes méthodes, cela renforce la confiance dans les valeurs de Shapley.

---
### **Évaluation Quantitative**
Utilisez des métriques quantitatives pour évaluer la performance des valeurs de Shapley. Voici quelques métriques couramment utilisées :

#### **Fidélité**
Mesurez la fidélité des explications en comparant les prédictions du modèle original avec les prédictions d'un modèle simplifié basé sur les valeurs de Shapley. Une bonne fidélité signifie que les explications sont proches des prédictions réelles du modèle.

---
---
## **9. Conclusion**
Les valeurs de Shapley permettent de **comprendre l'importance relative** de chaque *feature* dans les prédictions d'un modèle, que ce soit en **régression** ou en **classification**. Elles sont particulièrement utiles pour :
* **L'interprétabilité** des modèles complexes (ex. : *Random Forest*, *XGBoost*).
* **L'identification des *features* les plus influentes**.
* **La détection de biais** ou de dépendances non linéaires.
* **La comparaison équitable** des contributions, contrairement à des méthodes comme les coefficients de régression (qui dépendent de l'échelle des *features*).