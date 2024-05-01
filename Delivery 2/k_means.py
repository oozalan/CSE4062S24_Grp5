import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from pandas.plotting import parallel_coordinates
import seaborn as sns
from sklearn.metrics import silhouette_score
import collections

palette = sns.color_palette("bright", 10)

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    '''Display a scatter plot on a factorial plane, one for each factorial plane'''

    # For each factorial plane
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # Initialise the matplotlib figure      
            fig = plt.figure(figsize=(7,6))
        
            # Display the points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # Display the labels on the points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # Define the limits of the chart
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # Display grid lines
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection of points (on PC{} and PC{})".format(d1+1, d2+1))
            #plt.show(block=False)

def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0, bottom=0.4)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', wrap=True)
    
    plt.show()

# Read the dataset
df = pd.read_csv("Dataset/data.csv")
df.rename(columns=lambda x: x.strip(), inplace=True)
df.drop(columns=["Net Income Flag"], inplace=True)

# Seperate the class label from other features
X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]

# Find the 10 most positively and negatively correlated features
positive_corr = X.corrwith(y).sort_values(ascending=False)[:5].index.tolist()
negative_corr = X.corrwith(y).sort_values()[:5].index.tolist()
corr = positive_corr + negative_corr

# Filter the X to have only most correlated features
X = X.filter(items=corr)

# Find the optimal number of clusters using "elbow method" = 10
kmeans_tests = [KMeans(n_clusters=i, init='random', n_init=10) for i in range(1, 51)]
score = [kmeans_tests[i].fit(X).score(X) for i in range(len(kmeans_tests))]

# Plot the curve
plt.plot(range(1, 51), score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

# Run the k-means algorithm
kmeans = KMeans(init='random', n_clusters=10, n_init=100)
kmeans.fit(X)
clusters = kmeans.predict(X)

# Print the number of instances in each cluster
counter = collections.Counter(clusters)
for i in range (0, len(counter)):
    print("Cluster " + str(i) +  ": " + str(counter.get(i)))

# Calculate the silhouette score
print("Silhouette score: ", silhouette_score(X, kmeans.fit_predict(X)))

# Create a PCA model to reduce the dataset to 2 dimensions for visualization
X_arr = X.to_numpy()
pca = PCA(n_components=2)
pca.fit(X_arr)
X_reduced = pca.transform(X_arr)
centres_reduced = pca.transform(kmeans.cluster_centers_)

display_factorial_planes(X_reduced, 2, pca, [(0,1)], illustrative_var = clusters, alpha = 0.8)
plt.scatter(centres_reduced[:, 0], centres_reduced[:, 1],
            marker='x', s=169, linewidths=3,
            color='r', zorder=10)
plt.show()

# Create a data frame containing our centroids
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
centroids['cluster'] = centroids.index

display_parallel_coordinates_centroids(centroids, 10)

