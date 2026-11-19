from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
documents = [
    "we are going to mysore",
    "mysore is a famous place",
    "we are going to famous place",
]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
word_tfidf = dict(zip(feature_names, tfidf_scores))
print("TF-IDF scores for each word:")
for word, score in word_tfidf.items():
    print(f"{word}: {score:.4f}")
sorted_word_tfidf = sorted(word_tfidf.items(), key=lambda 
item: item[1], reverse=True)
print("\nWords with the highest TF-IDF values:") 
for word, score in sorted_word_tfidf: 
    print(f"{word}: {score:.4f}")