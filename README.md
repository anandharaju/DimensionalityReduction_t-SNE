# DimensionalityReduction_t-SNE
t-Distributed Stochastic Neighbor Embedding (t-SNE) is a probability based non-linear technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets. It is extensively applied in image processing, NLP, genomic data and speech processing. To keep things simple, here’s a brief overview of working of t-SNE:  The algorithms starts by calculating the probability of similarity of points in high-dimensional space and calculating the probability of similarity of points in the corresponding low-dimensional space. The similarity of points is calculated as the conditional probability that a point A would choose point B as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian (normal distribution) centered at A. It then tries to minimize the difference between these conditional probabilities (or similarities) in higher-dimensional and lower-dimensional space for a perfect representation of data points in lower-dimensional space. To measure the minimization of the sum of difference of conditional probability t-SNE minimizes the sum of Kullback-Leibler divergence of overall data points using a gradient descent method. Note Kullback-Leibler divergence or KL divergence is is a measure of how one probability distribution diverges from a second, expected probability distribution.  Those who are interested in knowing the detailed working of an algorithm can refer to this research paper.  In simpler terms, t-Distributed stochastic neighbor embedding (t-SNE) minimizes the divergence between two distributions: a distribution that measures pairwise similarities of the input objects and a distribution that measures pairwise similarities of the corresponding low-dimensional points in the embedding.  In this way, t-SNE maps the multi-dimensional data to a lower dimensional space and attempts to find patterns in the data by identifying observed clusters based on similarity of data points with multiple features. However, after this process, the input features are no longer identifiable, and you cannot make any inference based only on the output of t-SNE. Hence it is mainly a data exploration and visualization technique.

Scikit-learn's documentation of t-SNE explicitly states that:
"It is highly recommended to use another dimensionality reduction method (e.g., PCA for dense data or TruncatedSVD for sparse data) to reduce the number of dimensions to a reasonable amount (e.g., 50) if the number of features is very high. This will suppress some noise and speed up the computation of pairwise distances between samples."
You’ll now take this recommendation to heart and actually, reduce the number of dimensions before feeding the data into the t-SNE algorithm. For this, you’ll use PCA again. You will first create a new dataset containing the fifty dimensions generated by the PCA reduction algorithm, then use this dataset to perform the t-SNE.

Notable Tuning parameters for t-SNE:

n_components (default: 2): Dimension of the embedded space.
perplexity (default: 30): The perplexity is related to the number of nearest neighbors that are used in other manifold learning algorithms. Consider selecting a value between 5 and 50.
early_exaggeration (default: 12.0): Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them.
learning_rate (default: 200.0): The learning rate for t-SNE is usually in the range (10.0, 1000.0).
n_iter (default: 1000): Maximum number of iterations for the optimization. Should be at least 250.
method (default: ‘barnes_hut’): Barnes-Hut approximation runs in O(NlogN) time. method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time.

Key differences between PCA and t-SNE:
1) t-SNE is computationally expensive and can take several hours on million-sample datasets where PCA will finish in seconds or minutes.
2) PCA it is a mathematical technique, but t-SNE is a probabilistic one.
3) Linear dimensionality reduction algorithms, like PCA, concentrate on placing dissimilar data points far apart in a lower dimension representation. But in order to represent high dimension data on low dimension, non-linear manifold, it is essential that similar data points must be represented close together, which is something t-SNE does not PCA.
4) Sometimes in t-SNE different runs with the same hyperparameters may produce different results hence multiple plots must be observed before making any assessment with t-SNE, while this is not the case with PCA.
5) Since PCA is a linear algorithm, it will not be able to interpret the complex polynomial relationship between features while t-SNE is made to capture exactly that.

Source: https://www.datacamp.com/community/tutorials/introduction-t-sne

Dataset Used:
https://github.com/zalandoresearch/fashion-mnist

The Fashion-MNIST dataset is a 28x28 grayscale image of 70,000 fashion products from 10 categories, with 7,000 images per category. The training set has 60,000 images, and the test set has 10,000 images. Fashion-MNIST is a replacement for the original MNIST dataset for producing better results, the image dimensions, training and test splits are similar to the original MNIST dataset. Similar to MNIST the Fashion-MNIST also consists of 10 labels, but instead of handwritten digits, you have 10 different labels of fashion accessories like sandals, shirt, trousers, etc.
