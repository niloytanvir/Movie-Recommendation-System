import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


df = pd.read_csv("./Dataset/imdb_dataset.csv")
# print (df.columns)

features = ['keywords', 'cast', 'genres', 'director']
# Create a column in DF which combines all selected features

# handling the NaN values by using fillna function
# stackoverflow
for feature in features:
    df[feature] = df[feature].fillna('')


def combine_features(row):
    try:
        return row['keywords'] + " "+row['cast']+" "+row["genres"]+" "+row["director"]
    except:
        print("Error:", row)


df["combined_features"] = df.apply(combine_features, axis=1)

# print ("Combined Features:", df["combined_features"].head())

# Create count matrix from this new combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

# Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = input("Enter the movie you like: ")

# Gets the index of the movie from the data frame
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))

# Sort the movies depending on there similarity score in desending order
sorted_similar_movies = sorted(
    similar_movies, key=lambda x: x[1], reverse=True)

# display all movies
i = 0
for get_movie in sorted_similar_movies:
    print(get_title_from_index(get_movie[0]))
    i = i+1
    if i > 50:
        break
