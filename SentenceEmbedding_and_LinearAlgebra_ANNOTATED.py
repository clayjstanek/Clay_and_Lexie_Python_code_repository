# ============================================================
# SentenceEmbedding_and_LinearAlgebra_ANNOTATED.py
#
# Heavily documented teaching script for Lexie.
#
# PURPOSE
# -------
# Demonstrate that:
#   1) sentences can be converted into vectors,
#   2) cosine similarity can compare those vectors,
#   3) PCA can provide a rough 2D geometric picture of sentence clusters.
#
# This is not meant to teach NLP in depth.
# It is meant to show a "wow" idea:
#
#     Once something is mapped into vector space,
#     the same linear algebra tools still apply.
#
# Part 1: TF-IDF embeddings
#   - simple
#   - interpretable
#   - easy to explain
#
# Part 2: Optional modern sentence embeddings
#   - more semantically powerful
#   - requires sentence-transformers
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# TfidfVectorizer turns text into a sparse matrix of weighted word features.
from sklearn.feature_extraction.text import TfidfVectorizer

# cosine_similarity computes pairwise cosine similarity efficiently.
from sklearn.metrics.pairwise import cosine_similarity

# PCA will reduce high-dimensional vectors to 2 dimensions for plotting.
from sklearn.decomposition import PCA

# Toggle this if sentence-transformers is installed and you want the more
# advanced semantic-embedding demo.
RUN_MODERN_EMBEDDING_DEMO = True


# ------------------------------------------------------------
# Sentences for the demo
# ------------------------------------------------------------
# We intentionally mix topics:
#   - animals
#   - finance
#   - food
#   - weather
# so that some groups should be more similar than others.
sentences = [
    "The cat sat on the mat.",
    "A kitten rested on the rug.",
    "The dog chased the ball.",
    "A puppy ran after a toy.",
    "The stock market fell sharply today.",
    "Investors reacted to a sudden market decline.",
    "I love eating pizza with mushrooms.",
    "She ordered a pepperoni pizza for dinner.",
    "The weather is warm and sunny today.",
    "It is a bright sunny afternoon outside."
]


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def print_top_pairs(sim_matrix, sentences, k=5):
    """
    Print the top-k most similar pairs of sentences.

    HOW IT WORKS
    ------------
    The similarity matrix is symmetric.
    So we only need to inspect pairs where j > i.
    """
    pairs = []
    n = len(sentences)

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sim_matrix[i, j], i, j))

    # Sort from highest similarity to lowest.
    pairs.sort(reverse=True)

    print("\nMost similar sentence pairs:\n")

    for sim, i, j in pairs[:k]:
        print(f"Similarity = {sim:.3f}")
        print(" ", sentences[i])
        print(" ", sentences[j])
        print()


def plot_similarity_heatmap(sim_matrix):
    """
    Plot the full pairwise cosine similarity matrix as a heatmap.

    INTERPRETATION
    --------------
    Bright values near 1 mean strong similarity.
    Darker values near 0 mean weaker similarity.
    The diagonal is always 1 because every sentence is identical to itself.
    """
    n = sim_matrix.shape[0]

    plt.figure(figsize=(7, 6))
    plt.imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.xticks(range(n), range(n))
    plt.yticks(range(n), range(n))
    plt.title("Sentence Similarity Heatmap")
    plt.show()


def plot_similarity_histogram(sim_matrix):
    """
    Plot a histogram of off-diagonal pairwise similarities.

    WHY OMIT THE DIAGONAL?
    ---------------------
    The diagonal entries are all exactly 1. Those are not informative because
    they compare each sentence to itself.
    """
    sims = []
    n = sim_matrix.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            sims.append(sim_matrix[i, j])

    plt.figure()
    plt.hist(sims, bins=12)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.title("Distribution of Sentence Similarity")
    plt.show()


def plot_pca_projection(vectors):
    """
    Reduce the high-dimensional sentence vectors to 2D with PCA and scatter plot them.

    IMPORTANT TEACHING POINT
    ------------------------
    PCA is not preserving every detail perfectly. It is giving us a rough 2D view
    of a higher-dimensional geometry.
    """
    pca = PCA(n_components=2)

    # Fit PCA and project all vectors into a 2D coordinate system.
    X2 = pca.fit_transform(vectors)

    plt.figure(figsize=(7, 6))
    plt.scatter(X2[:, 0], X2[:, 1], s=80)

    # Label each point by sentence index so we can connect the plot to the sentence list.
    for i in range(len(X2)):
        plt.text(X2[i, 0], X2[i, 1], str(i))

    plt.title("Sentence Vectors Projected into 2D (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


# ------------------------------------------------------------
# TF-IDF Demonstration
# ------------------------------------------------------------
def tfidf_demo():
    """
    Demonstrate a simple text-to-vector method: TF-IDF.

    BIG IDEA
    --------
    We are converting each sentence into a numeric vector whose entries reflect
    important vocabulary terms.
    """
    print("\n==============================")
    print("TF-IDF Sentence Embeddings")
    print("==============================\n")

    # stop_words="english" removes very common words like "the", "is", "a"
    # so that more meaningful vocabulary terms dominate the vectors.
    vectorizer = TfidfVectorizer(stop_words="english")

    # fit_transform learns the vocabulary and transforms the text into a matrix.
    # We convert to a dense array so it prints and plots more easily.
    X = vectorizer.fit_transform(sentences).toarray()

    print("Vector matrix shape:", X.shape)
    print("(rows = sentences, columns = vocabulary terms)\n")

    # Compute all pairwise cosine similarities.
    sim_matrix = cosine_similarity(X)

    np.set_printoptions(precision=3, suppress=True)

    print("Cosine similarity matrix:\n")
    print(sim_matrix)

    print_top_pairs(sim_matrix, sentences)
    plot_similarity_heatmap(sim_matrix)
    plot_similarity_histogram(sim_matrix)
    plot_pca_projection(X)

    print("\nTeaching note:")
    print("TF-IDF captures shared important words.")
    print("It is useful and interpretable, but not deeply semantic.")


# ------------------------------------------------------------
# Optional: Modern Sentence Embeddings
# ------------------------------------------------------------
def modern_embedding_demo():
    """
    Optional higher-powered semantic embedding demo.

    This uses a pretrained neural sentence embedding model.
    If installed, it often gives a stronger "wow" because semantically similar
    sentences can be close even if they do not share many exact words.
    """
    print("\n==============================")
    print("Modern Sentence Embeddings")
    print("==============================\n")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        print("sentence-transformers not installed.")
        print("Run: pip install sentence-transformers")
        return

    # This model returns a dense embedding vector for each sentence.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode each sentence into a dense vector.
    embeddings = model.encode(sentences)

    sim_matrix = cosine_similarity(embeddings)

    np.set_printoptions(precision=3, suppress=True)

    print("Cosine similarity matrix:\n")
    print(sim_matrix)

    print_top_pairs(sim_matrix, sentences)
    plot_similarity_heatmap(sim_matrix)
    plot_similarity_histogram(sim_matrix)
    plot_pca_projection(embeddings)

    print("\nTeaching note:")
    print("These embeddings are designed to reflect semantic similarity more strongly.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    """
    Main execution path for the lesson.
    """
    print("\nMapping language into vectors...\n")

    # Print sentence list with numeric IDs.
    for i, s in enumerate(sentences):
        print(i, ":", s)

    # Always run the interpretable TF-IDF demo first.
    tfidf_demo()

    # Optionally run the more advanced semantic embedding demo.
    if RUN_MODERN_EMBEDDING_DEMO:
        modern_embedding_demo()

    print("\nKey idea:")
    print("Once sentences become vectors, we can use linear algebra")
    print("to measure similarity and structure in language.")


if __name__ == "__main__":
    main()
