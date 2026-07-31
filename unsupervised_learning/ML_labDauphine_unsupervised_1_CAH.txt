# The packages

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

i = np.array([8, 5])
j = np.array([2, 7])

print(np.sqrt(np.sum((i - j)**2)))

# Verification:
print(((8 - 2)**2 + (5 - 7)**2)**0.5)

a = np.array([8, 5])
b = np.array([2, 7])
c = np.array([3, 6])

cluster_1 = np.array([a, b])
print(f"cluster_1: \n{cluster_1}")

cluster_2 = np.array([c])
print(f"cluster_2: \n{cluster_2}")

for i in cluster_1:
    for j in cluster_2:
        print(f"Distance between {i} and {j}: {np.sqrt(np.sum((i - j)**2))}")

def cluster_distance(c1, c2):
    '''Using the euclidean distance to calculate the distance between two clusters.'''
    min_dist = float('inf')
    for i in c1:
        for j in c2:
            dx = i[0] - j[0]
            dy = i[1] - j[1]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
    return min_dist


cluster_distance(cluster_1, cluster_2)

def print_simple_dendrogram():
    print("A   B        C   D   E")
    print("|   |        |   |   |")
    print(" \\ /         \\ /    |      <- 1.0 et 1.4")
    print("  AB          CD     |")
    print("          _____|_____/      <- 1.4")
    print("         |           |")
    print("         |          CDE")
    print("         \\___________/      <- 2.8 (final merge)")

print_simple_dendrogram()


import numpy as np

# Dictionnaire of the points with labels
points = {
    "A": np.array([1, 1]),
    "B": np.array([2, 1]),
    "C": np.array([4, 3]),
    "D": np.array([5, 4]),
    "E": np.array([3, 4])
}
points

# Initialize the clusters:
clusters = {f"C{i+1}": {label} for i, label in enumerate(points.keys())}

labels = list(points.keys())
labels

# Convert to array for easier indexing
labels = list(points.keys())
X = np.array([points[label] for label in labels])
print(X)

n = len(labels)

# Initialize the matrix of distances (n x n)
dist_matrix = np.zeros((n, n))

# Calculate the euclidean distances between all the points
for i in range(n):
    for j in range(n):
        if i != j:
            pi = points[labels[i]]
            pj = points[labels[j]]
            dist = np.sqrt(np.sum((pi - pj) ** 2))
            dist_matrix[i, j] = dist


print("Distance matrix (symmetric):")
print("   ", "  ".join(labels))
for i, row in enumerate(dist_matrix):
    print(labels[i], row)


from scipy.spatial.distance import euclidean

# Step 0: Initialization — Each point is its own cluster
clusters = {i: [i] for i in range(len(X))}
history = []

def cluster_distance(c1, c2):
    # Single linkage: min distance between any point in cluster 1 and any point in cluster 2
    # we use Euclidean distance
    return min(np.sqrt(np.sum((X[i] - X[j])**2)) for i in c1 for j in c2)

step = 1
while len(clusters) > 1:
    # Find the two closest clusters
    pairs = [(i, j, cluster_distance(clusters[i], clusters[j]))
             for i in clusters for j in clusters if i < j]
    i_min, j_min, dist_min = min(pairs, key=lambda x: x[2])

    # Explanation step-by-step
    print(f"> Step {step}")
    print(f"> Merge clusters: {clusters[i_min]} + {clusters[j_min]} at distance {dist_min:.2f}")

    # Save history for plotting later
    history.append(dist_min)

    # Merge clusters
    new_cluster = clusters[i_min] + clusters[j_min]
    new_index = max(clusters) + 1
    clusters[new_index] = new_cluster
    del clusters[i_min]
    del clusters[j_min]

    step += 1

# === PLOT DISTANCE EVOLUTION ===

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(history) + 1), history, marker='o', linestyle='-')
plt.title("Evolution of Merge Distances (Single Linkage)")
plt.xlabel("Merge Step")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

# === STANDARD LINKAGE & DENDROGRAM ===

Z = linkage(X, method='single', metric='euclidean')

plt.figure(figsize=(8, 4))
dendrogram(Z, labels=labels)
plt.title("Dendrogram (Single Linkage)")
plt.xlabel("Data points")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

## HAC using Scipy

# R code: 
# cluster.colors <- brewer.pal(8,"Dark2") 
# blobs <- read.table(file="Data/blobs.txt", header=F, sep=",") ggplot(blobs, aes(x=V1, y=V2)) + geom_point()

# Install and import necessary libraries
# !pip install pandas matplotlib seaborn palettable
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from palettable.colorbrewer.qualitative import Dark2_8

# Define the cluster colors using the Dark2 palette
cluster_colors = Dark2_8.hex_colors

# Read the data from 'blobs.txt' into a pandas DataFrame
# The file has no header and is comma-separated
path_name = '/Users/davidtbo/Library/Mobile Documents/com~apple~CloudDocs/data/external'
blobs = pd.read_csv(filepath_or_buffer=os.path.join(path_name,'blobs.txt'), header=None)

# Create a scatter plot using seaborn
# V1 (first column) is mapped to the x-axis, V2 (second column) is mapped to the y-axis
sns.scatterplot(x=blobs.iloc[:, 0], y=blobs.iloc[:, 1])

# Display the plot
plt.show()

# R code: 
# dend <- hclust(dist(blobs[,1:2], method="euclidean"), method="single")

# Minimum jump clustering
# Hierarchical clustering using the minimum jump method
# Calculates the Euclidean distance between data points (first two columns of 'blobs')
# Performs hierarchical clustering using the single linkage method ('minimum jump')

# 1) On calcule les distances euclidienne entre les points de données (premières deux colonnes de 'blobs')
# 2) On effectue le clustering hiérarchique en utilisant la méthode de Single linkage (minimum jump)

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

# Calculate the pairwise euclidean distance (matrix)

dist_matrix = ssd.pdist(blobs.iloc[:, 0:2], metric='euclidean')

# Perform single linkage hierarchical clustering (see the course: ML_coursDauphine_unsupervised_1_CAH_Kmeans.pdf) 

# dend = sch.linkage(dist_matrix, method='single')
# dend = sch.linkage(dist_matrix, method='complete')
dend = sch.linkage(dist_matrix, method='average')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
sch.dendrogram(dend)
plt.title('Dendrogram (Minimum Jump Clustering)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# prompt: plot(dend$height, type="b")

# Plot the evolution of the aggregation criterion 
# (montre l'évolution des distances de fusion)
# The idea is to use the "break" (elbow, jump) in the aggregation distance curve:
# At first, the distances are small (merging of nearby clusters).
# When we start forcing the merger of very distant clusters, the distance suddenly increases.
# The goal is to cut the dendrogram just before this big jump, which gives a more "natural" number of clusters.
# dend[:, 2] contains the linkage distances (aggregation criterion)
plt.plot(dend[:, 2], marker='o', linestyle='-') # Use marker='o' for points and linestyle='-' for lines
plt.xlabel("Step")
plt.ylabel("Aggregation Criterion")
plt.title("Evolution of Aggregation Criterion")
plt.show()

from scipy.cluster.hierarchy import fcluster

# Cut the dendrogram to get 3 clusters

clusters = fcluster(dend, 3, criterion='maxclust')

# To get a summary similar to `summary(as.factor(clusters))` in R,
# we can use pandas value_counts.
print("Cluster distribution:")
print(pd.Series(clusters).value_counts().sort_index())

# The `order_clusters_as_data = F` in R's `cutree` affects how the clusters are
# ordered in the returned vector. By default, `fcluster` returns the cluster
# assignments in the order of the original data points. This behavior is similar
# to `order_clusters_as_data = TRUE` in R.
# If you needed the clusters ordered based on the dendrogram leaves order (which is less common
# when just getting flat clusters), you would need to reorder the original data
# based on the dendrogram's leaf order before calling fcluster or reorder
# the resulting cluster array. However, for summarizing the cluster distribution,
# the order doesn't matter, and the default behavior of `fcluster` is usually what's desired
# for assigning cluster labels back to the original data points.

# R code: 
# dend <- color_branches(as.dendrogram(dend), clusters=clusters, col=cluster.colors[1:3]) 
# clusters <- cutree(dend, 3, order_clusters_as_data = T) 
# plot.list <-list(ggplot(as.ggdend(dend)),ggplot(blobs, aes(V1,V2)) + geom_point(col=cluster.colors[clusters], size=0.2)) 
# ggmatrix(plot.list, nrow=1, ncol=2, showXAxisPlotLabels = F, showYAxisPlotLabels = F, xAxisLabels=c("dendrogram", "scatter plot")) + theme_bw()

# !pip install plotnine mizani
# Importing necessary libraries
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import pandas as pd # Assuming blobs is a pandas DataFrame and clusters is a pandas Series or numpy array

# Assuming 'dend' is the linkage matrix from sch.linkage
# Assuming 'blobs' is a pandas DataFrame with the original data
# Assuming 'clusters' is a numpy array or pandas Series containing the cluster assignments (1, 2, or 3)

# Create a figure and a set of subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the dendrogram in the first subplot
# We don't directly color branches based on fcluster results here as it's complex with scipy
sch.dendrogram(dend, ax=axes[0])
axes[0].set_title('Dendrogram')
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Distance')

# Define colors for the scatter plot based on the number of clusters
# Using a colormap and mapping cluster labels to colors
num_clusters = len(set(clusters))
cmap = plt.colormaps.get_cmap('Dark2') # Using the recommended method to get a colormap
colors = [cmap(i / (num_clusters - 1)) for i in range(num_clusters)] if num_clusters > 1 else [cmap(0)]

# Plot the scatter plot with colored points based on cluster assignments in the second subplot
# Need to map cluster labels (1, 2, 3, ...) to colormap indices (0, 1, 2, ...)
# Assuming clusters are 1-indexed, subtract 1 for 0-indexed colormap access
scatter = axes[1].scatter(blobs.iloc[:, 0], blobs.iloc[:, 1], c=[colors[c-1] for c in clusters], s=5)
axes[1].set_title('Scatter Plot with Clusters')
axes[1].set_xlabel(blobs.columns[0]) # Use actual column names if available, or generic 'V1'
axes[1].set_ylabel(blobs.columns[1]) # Use actual column names if available, or generic 'V2'


# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()

# Note: The R code's `color_branches` functionality which colors the dendrogram branches
# based on the flat clustering result is not directly replicated here as it requires
# significant manipulation of the dendrogram plotting output which is beyond
# a simple conversion. The focus is on displaying the dendrogram and the
# clustered data side-by-side.

# R code: 
# table(clusters, blobs$V3)

# Assuming 'clusters' is a numpy array or pandas Series
# Assuming 'blobs' is a pandas DataFrame and V3 refers to the 3rd column (index 2)

# To perform the equivalent of R's `table(clusters, blobs$V3)`
# we can use pandas `crosstab` function.
# `crosstab` computes a frequency table of two (or more) factors.
# The first argument `index` corresponds to the first factor (clusters)
# The second argument `columns` corresponds to the second factor (blobs['V3'])

# Ensure that the lengths of 'clusters' and 'blobs' are compatible.
# The number of cluster assignments should match the number of rows in blobs.
if len(clusters) == len(blobs):
    print("\nFrequency table of clusters vs. original category (blobs column V3):")
    # Use blobs.iloc[:, 2] to access the third column (V3)
    contingency_table = pd.crosstab(clusters, blobs.iloc[:, 2])
    print(contingency_table)
else:
    print("Error: The number of cluster assignments does not match the number of data points in blobs.")



# R code: 
# dend <- hclust(dist(blobs[,1:2], method="euclidean"), method="ward.D2")

# Perform Ward's method hierarchical clustering
# Calculates the Euclidean distance between data points (first two columns of 'blobs')
# Performs hierarchical clustering using Ward's method
# `blobs.iloc[:, 0:2]` selects the first two columns of the DataFrame 'blobs'
# `metric='euclidean'` specifies the distance metric as Euclidean
# `method='ward'` specifies Ward's linkage method


# Calculate the pairwise euclidean distance (matrix)

dist_matrix = ssd.pdist(blobs.iloc[:, 0:2], metric='euclidean')

dend_ward = sch.linkage(dist_matrix, method='ward')

# Plot the dendrogram for Ward's method
plt.figure(figsize=(10, 7))
sch.dendrogram(dend_ward)
plt.title('Dendrogram (Ward\'s Method)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Plot the evolution of the aggregation criterion for Ward's method
plt.plot(dend_ward[:, 2], marker='o', linestyle='-')
plt.xlabel("Step")
plt.ylabel("Aggregation Criterion (Ward's)")
plt.title("Evolution of Aggregation Criterion (Ward's Method)")
plt.show()

# prompt: clusters <- cutree(dend, 3, order_clusters_as_data = F)
# summary(as.factor(clusters))

# Based on the dendrogram and the evolution of the aggregation criterion plot for Ward's method,
# visually identify the point where the increase in the criterion is largest.
# The comment "Choix d’un découpage à 3 classes" suggests that visually,
# there is a large jump in the criterion for Ward's method that would lead to choosing 3 classes.

# The equivalent R code `cutree(dend, 3, order_clusters_as_data = F)` for Ward's method
# using scipy's `fcluster` function.
# Cut the dendrogram `dend_ward` to get 3 clusters.
clusters_ward = fcluster(dend_ward, 3, criterion='maxclust')

# To get a summary similar to `summary(as.factor(clusters))` in R for Ward's method,
# use pandas value_counts.
print("\nCluster distribution (Ward's Method):")
print(pd.Series(clusters_ward).value_counts().sort_index())

# !pip install plotnine mizani
from plotnine import ggplot, aes, geom_point, theme_bw
from mizani.formatters import percent_format
from plotnine.guides import guide_legend
from plotnine.scales import scale_x_continuous, scale_y_continuous
from plotnine.labels import labs

# Import `ggmatrix` if needed, though a simple subplot approach is used below
# !pip install ggmatrix # Install if you want to use ggmatrix (less common in Python plotnine context)
# from ggmatrix import ggmatrix # Import if installed

# Import necessary libraries for plotting the dendrogram using matplotlib
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import numpy as np

# Assuming 'dend' is the linkage matrix from sch.linkage (from the single linkage method)
# Assuming 'clusters' is a numpy array or pandas Series containing the cluster assignments (1, 2, or 3) from single linkage
# Assuming 'blobs' is a pandas DataFrame with the original data

# Define colors for the scatter plot based on the number of clusters
# Using the cluster_colors defined earlier (Dark2_8 palette)
# Ensure the number of colors is sufficient for the number of clusters
num_clusters = len(set(clusters))
if num_clusters > len(cluster_colors):
    print(f"Warning: Not enough colors defined for {num_clusters} clusters. Using repeated colors.")
    colors_for_scatter = [cluster_colors[c % len(cluster_colors)] for c in clusters]
else:
    # Map cluster labels (1, 2, 3, ...) to the defined colors (index 0, 1, 2, ...)
    colors_for_scatter = [cluster_colors[c-1] for c in clusters]


# Create a figure and a set of subplots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(15, 7)) # Adjust figsize as needed

# Plot the dendrogram in the first subplot using matplotlib
# To color branches by flat clusters, this requires more advanced manipulation of the dendrogram object,
# which is not as straightforward as in R's `color_branches`. A common approach in matplotlib
# is to draw the dendrogram first and then potentially add colored lines/patches afterwards,
# or use the `color_threshold` argument if cutting by distance, but not directly by maxclust.
# For simplicity here, we plot the standard dendrogram.
sch.dendrogram(dend, ax=axes[0])
axes[0].set_title('Dendrogram')
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Distance')

# Plot the scatter plot with colored points based on cluster assignments in the second subplot using matplotlib
# Use the colors_for_scatter list generated based on cluster assignments
axes[1].scatter(blobs.iloc[:, 0], blobs.iloc[:, 1], c=colors_for_scatter, s=10) # Adjust size 's' as needed
axes[1].set_title('Scatter Plot with Clusters')
axes[1].set_xlabel(blobs.columns[0])
axes[1].set_ylabel(blobs.columns[1])

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()

# Note: Directly replicating `ggmatrix` and `as.ggdend` from R in Python using plotnine
# and matplotlib is complex. The approach above uses matplotlib for both plots within
# a single figure's subplots, providing a similar side-by-side visualization.
# If you specifically need plotnine for the scatter plot, you would create a plotnine
# object for the scatter plot and display it separately or attempt to combine it
# using more advanced methods or libraries that support combining matplotlib and plotnine plots.
# The matplotlib approach for both is generally more direct when showing the dendrogram alongside.

# If you want to use plotnine for the scatter plot:
# scatter_plot = (
#     ggplot(blobs, aes(x=blobs.columns[0], y=blobs.columns[1], color=clusters.astype(str))) # Use string for discrete color
#     + geom_point(size=0.2)
#     + labs(x="V1", y="V2", color="Cluster")
#     + theme_bw()
# )

# And then display the matplotlib dendrogram and the plotnine scatter plot.
# Combining them into a single figure as `ggmatrix` does in R is not a standard
# feature of plotnine/matplotlib without significant custom code or using a
# specialized library if one exists. The matplotlib subplot approach shown above
# is the most common way to achieve side-by-side plots in Python.

# prompt: table(clusters, blobs$V3)

# The R code `table(clusters, blobs$V3)` creates a contingency table
# showing the counts of observations for each combination of `clusters`
# and the values in the third column of `blobs` (indexed as 2 in pandas).

# This was already implemented in the preceding code block.
# To reiterate the code for clarity:

# Ensure that the lengths of 'clusters' and 'blobs' are compatible.
if len(clusters) == len(blobs):
    print("\nFrequency table of clusters vs. original category (blobs column V3):")
    # Use blobs.iloc[:, 2] to access the third column (V3)
    contingency_table = pd.crosstab(clusters, blobs.iloc[:, 2])
    print(contingency_table)
else:
    print("Error: The number of cluster assignments does not match the number of data points in blobs.")

import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the IRIS dataset
# head(iris) equivalent in pandas
iris = sns.load_dataset('iris')
print(iris.head())

# Normalization (zero mean, unit variance)
# Equivalent to iris.norm <- data.frame(sapply(iris[,1:4], scale)) in R
scaler = StandardScaler()
iris_norm_data = scaler.fit_transform(iris.iloc[:, 0:4])
iris_norm = pd.DataFrame(iris_norm_data, columns=iris.columns[0:4])

# Add the 'Species' column back to the normalized dataframe
# Equivalent to iris.norm$Species <- iris$Species
iris_norm['species'] = iris['species']

# Scatter plot matrices and distribution of classes by variable
# Equivalent to ggpairs(iris, columns=1:4, aes(color=Species)) using seaborn's pairplot
# Note: ggpairs from R's GGally is more comprehensive, pairplot is a common alternative in Python
sns.pairplot(iris, hue="species")
plt.suptitle("Scatter plot matrix and distribution of classes by variable for IRIS dataset", y=1.02) # Add a title
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import numpy as np

# Hierarchical Classification with Ward's method
# Equivalent to dend <- hclust(dist(iris.norm[,1:4], method="euclidean"), method="ward.D2") in R
linked = linkage(iris_norm.iloc[:, 0:4], method='ward')

# Dendrogram
# Equivalent to plot(dend) in R
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title("Dendrogram of IRIS dataset")
plt.xlabel("Data points")
plt.ylabel("Distance")
plt.show()

# Evolution of the aggregation criterion
# Equivalent to plot(dend$height, type="b") in R
plt.figure(figsize=(10, 7))
plt.plot(linked[:, 2], marker='o')
plt.title("Evolution of the aggregation criterion")
plt.xlabel("Number of merges")
plt.ylabel("Distance")
plt.show()


# Choosing a cut at 5 classes
# Equivalent to clusters <- cutree(dend, 5, order_clusters_as_data = F) in R
# Note: cutree in scipy does not have an equivalent to order_clusters_as_data = F,
# the clusters are assigned in the order of the original data.
clusters = cut_tree(linked, n_clusters=5).flatten()

# Summary of the number of elements in each cluster
# Equivalent to summary(as.factor(clusters)) in R
unique_clusters, counts = np.unique(clusters, return_counts=True)
print("Cluster distribution:")
for cluster_id, count in zip(unique_clusters, counts):
    print(f"Cluster {cluster_id}: {count}")

# Dendrogram with the partitioning and obtained clustering
# Equivalent to dend <- color_branches(as.dendrogram(dend), clusters=clusters, col=cluster.colors[1:5]) in R
# and the subsequent plotting code.
# Coloring branches in matplotlib dendrogram is more involved than in R's dendextend.
# We'll replot the dendrogram and potentially add labels/colors manually if needed.
plt.figure(figsize=(10, 7))
dendrogram(
    linked,
    leaf_rotation=90.,
    leaf_font_size=8.,
    labels=clusters # Using cluster labels as leaf labels for visualization
)
plt.title("Dendrogram of IRIS dataset with 5 Clusters")
plt.xlabel("Data points (colored by cluster)")
plt.ylabel("Distance")
plt.show()

# Scatter plot with clustering results
# Equivalent to ggplot(iris, aes(Petal.Length, Petal.Width)) + geom_point(col=cluster.colors[clusters], size=1))
plt.figure(figsize=(8, 6))
scatter = plt.scatter(iris['petal_length'], iris['petal_width'], c=clusters, cmap='viridis', s=10)
plt.title("Petal Length vs Petal Width colored by Cluster")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.colorbar(scatter, label='Cluster ID')
plt.show()

# Creating a combined plot (Equivalent to ggmatrix part)
# This requires more custom plotting. We'll show the two plots separately as direct equivalent of ggmatrix isn't straightforward in basic matplotlib/seaborn.

# Confusion matrix
# Equivalent to table(clusters, iris$Species) in R
from sklearn.metrics import confusion_matrix
# Need to map the species names to numerical labels to compare with cluster IDs
species_map = {species: i for i, species in enumerate(iris['species'].unique())}
true_labels = iris['species'].map(species_map)

conf_matrix = confusion_matrix(true_labels, clusters)
print("\nConfusion Matrix (True Species vs Clusters):")
conf_matrix

# For a more detailed comparison, you might want to see how each cluster relates to the original species.
# Note that cluster IDs from cut_tree are arbitrary and don't necessarily correspond to the original species labels.
# You would typically assign cluster labels to the majority species in each cluster for interpretation.

