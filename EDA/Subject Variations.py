import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

if 'question' not in df.columns or 'subject_name' not in df.columns:
    raise ValueError("DataFrame must contain 'question' and 'subject_name' columns")

# Vectorize the questions
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['question'])

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X.toarray())

# Perform K-Means clustering
num_clusters = len(df['subject_name'].unique())  # Number of clusters equals the number of unique subjects
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(X_tsne)

# Create a DataFrame for plotting
plot_df = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
plot_df['subject_name'] = df['subject_name'].values
plot_df['cluster'] = clusters

# Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(plot_df['Dim1'], plot_df['Dim2'], c=plot_df['cluster'], cmap='tab20', alpha=0.6, s=50)
plt.title('t-SNE of Questions with K-Means Clustering')
plt.xlabel('Topic')
plt.ylabel('Subject')

# Add legend
handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
plt.legend(handles, df['subject_name'].unique(), title="Subjects", bbox_to_anchor=(1.05, 1), loc='best')

plt.grid(True)
plt.tight_layout()
plt.show()