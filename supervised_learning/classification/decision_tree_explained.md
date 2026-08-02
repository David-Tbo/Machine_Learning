
## 1. Dataset split

Il est nécessaire d'avoir une fonction qui sépare le dataset en deux parties (arbre gauche et arbre droit) selon un critère numérique et un seuil.

```python
def split_tree(tree, criteria, threshold):
    left_tree = tree[tree[criteria] < threshold]
    right_tree = tree[tree[criteria] >= threshold]
    return left_tree, right_tree
```

## 2. L'entropie d'un noeud (classification binaire)

L'entropie $H$ mesure l'incertitude (impureté) d'un noeud.
* Forte incertitude si $p\approx 0.5$
$$p \approx 0.5 \Rightarrow H(p) \approx \infty$$
* Faible incertitude si $p\approx 0,  \text{ou } p \approx 1$
$$p \approx 0 \text{ ou } 1\Rightarrow H(p) \approx 0$$


```python
def entropy_node(tree, binary_target):
    n = tree.shape[0]
    if n == 0:
        return 0
    p = tree[binary_target].mean()
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
```

# 3. Entropy after split

```python
def entropy_split(left_tree, right_tree, binary_target):
    '''Entropy after split'''

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
```

# 4. Information gain

def information_gain(tree, criteria, threshold, binary_target):
    '''Maximize the decrease in entropy after the split'''

    parent_entropy = entropy_node(tree, binary_target)

    left_tree, right_tree = split_tree(tree, criteria, threshold)
    split_entropy = entropy_split(left_tree, right_tree, binary_target)

    gain = parent_entropy - split_entropy

    return gain


# 5. Find the best split

def find_best_threshold(tree, criteria, binary_target):
    '''Find the threshold that maximizes the information gain.
    Simple approach that iterates through all unique values of the criteria 
    and calculates the gain for each threshold. 
    '''
    thresholds = tree[criteria].unique()
    best_gain = -np.inf
    best_threshold = None

    for threshold in thresholds:
        gain = information_gain(tree, criteria, threshold, binary_target)
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain

# Load the dataset

```python
import os

data_path = '/Users/davidtbo/Library/Mobile Documents/com~apple~CloudDocs/data/external'
file_path = os.path.join(data_path, 'diabetes.csv')

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Fichier introuvable : {file_path}")

df = pd.read_csv(file_path)

# Standardize column names
df.columns = df.columns.str.lower()

# Drop duplicates
df = df.drop_duplicates()

# # Quick diagnostic
# print(df.info())
# print(df.isna().mean())

df.head()
# 0. Train-test split

X = df.drop(columns='outcome')
y = df['outcome']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
df_train = pd.concat([X_train, y_train], axis=1)
df_train.head()
# Test the function on the training set
find_best_threshold(df_train, "glucose", "outcome")
# Find the best threshold for each feature
for col in df_train.columns[:-1]:  # Exclude the target column
    best_threshold, best_gain = find_best_threshold(df_train, col, "outcome")
    print(f"Best threshold for {col}: {best_threshold} with gain: {best_gain:.4f}")
# The best gain is for glucose at threshold 155. Let's split the tree on this threshold and calculate the entropy after the split.
left_tree, right_tree = split_tree(df_train, "glucose", 155)
Note: We can improve the previous loop:
# Find the best threshold for each feature

best_gain = -np.inf
best_threshold = None

for col in df_train.columns[:-1]:  # Exclude the target column
    threshold, gain = find_best_threshold(df_train, col, "outcome")
    if gain > best_gain:
        best_gain = gain
        best_threshold = threshold
print(f"Best threshold: {best_threshold} with gain: {best_gain:.4f}")
# And we can find the best threshold for each feature in the left tree (glucose < 155)
for col in left_tree.columns[:-1]:  # Exclude the target column
    best_threshold, best_gain = find_best_threshold(left_tree, col, "outcome")
    print(f"Best threshold for {col}: {best_threshold} with gain: {best_gain:.4f}")
# For the left tree, the best gain is for bmi at threshold 26.8. 
# Let's split the tree on this threshold and calculate the entropy after the split.
```

left_subtree, right_subtree = split_tree(left_tree, "bmi", 26.8)
1. This is a manual process (we decide left and right), and we could continue to split till the last point.right_tree. 

So we need to make this process automated and give to it a stopping criterion (e.g., max depth, min samples per leaf, etc.)
2. **Let's build a recursive function to build the decision tree.**  

We will use a simple stopping criterion: if the gain is less than a certain threshold, we will stop splitting.
__build_tree(data)__  
* if data is pure enough or too small:  
    * return a leaf  

* find the best split
* split data into left_data and right_data  

* build the sub left tree with left_data  
* build the sub right tree with right_data  

* return a node containing:  
    * variable  
    * threshold  
    * left_subtree  
    * right_subtree  
The key point is:

* **left_subtree = build_tree(left_data)**

* **right_subtree = build_tree(right_data)**
The function calls itself with smaller datasets.  
3. We need a structure (dictionary) to store:

* The tree
node = {
    "type": "node",
    "feature": "glucose",
    "threshold": 120,
    "left": left_subtree,
    "right": right_subtree,
}
* And a leaf
leaf = {
    "type": "leaf",
    "prediction": 1
}
4. We need to define a **stopping criterion**.  

It ensures that the tree does not grow indefinitely.
Simple examples:
* if entropy_node(tree, target) == 0:
    * return leaf

* if depth >= max_depth:
    * return leaf

* if len(tree) < min_samples_split:
    * return leaf

* if best_gain <= 0:
    * return leaf
The prediction of a leaf is often the majority class of the samples in that leaf.  

In this case, since we are working with a binary classification problem, the prediction would be 1:
>prediction = int(tree[target].mean() >= 0.5)
5. **Skeleton version**
def majority_class(tree, binary_target):
    return int(tree[binary_target].mean() >= 0.5)


def build_tree(tree, features, binary_target, depth=0, max_depth=3, min_samples_split=5):
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
6. Intuitive lecture. 

When you call:
> tree_model = build_tree(df_train, features, "target", max_depth=3)
Python does:
build_tree(racine)
* build_tree(sous-table gauche)
    * build_tree(sous-table gauche-gauche)
    * build_tree(sous-table gauche-droite)

* build_tree(sous-table droite)
    * build_tree(sous-table droite-gauche)
    * build_tree(sous-table droite-droite)
This is the manual process, automotized.
7. Then, prediction

Once the tree has been built, predire revient à descendre dans le dictionnaire:
def predict_one(row, tree_model):
    if tree_model["type"] == "leaf":
        return tree_model["prediction"]

    feature = tree_model["feature"]
    threshold = tree_model["threshold"]

    if row[feature] < threshold:
        return predict_one(row, tree_model["left"])
    else:
        return predict_one(row, tree_model["right"])
Là encore, c’est récursif : tant qu’on n’est pas sur une feuille, on descend à gauche ou à droite.  

D'ailleurs on commence aussi par un test d'arrêt si feuille.

Petit détail : dans ton fichier actuel, tu charges load_breast_cancer(), pas diabetes. Et il faudra convertir X_train, y_train en DataFrame avec noms de colonnes + colonne target pour utiliser tes fonctions actuelles basées sur tree[criteria].
**Concrete case**
Cas concret. L’idée est :

* Tu concatènes X_train et y_train.
* Tu construis l’arbre avec build_tree(...).
* Pour prédire une ligne, predict_one(...) descend dans l’arbre jusqu’à une feuille.

Exemple avec un DataFrame df_train qui contient outcome :
# If X_train is already a DataFrame
df_train = X_train.copy()
df_train["outcome"] = y_train

features = [col for col in df_train.columns if col != "outcome"]

tree_model = build_tree(
    tree=df_train,
    features=features,
    binary_target="outcome",
    max_depth=3,
    min_samples_split=10
)
tree_model
Prediction on row test data:
tree_model['feature']
row = X_test.iloc[0]
row
prediction = predict_one(row, tree_model)
prediction
Si l’arbre dit :
glucose < 155 ?
Alors :

* oui => aller dans left
* non => aller dans right

Puis il recommence sur le sous-arbre suivant, par exemple :
bmi < 31.4 ?
Et dès qu’il tombe sur :
{"type": "leaf", "prediction": 1}
il retourne 1.
Pour prédire tout X_test :
y_pred = X_test.apply(lambda row: predict_one(row, tree_model), axis=1)
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy:.4f}")
# END