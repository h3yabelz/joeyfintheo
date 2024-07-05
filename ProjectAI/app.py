from flask import Flask, request, render_template
from pyngrok import ngrok
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

app = Flask(__name__)

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# Path relatif atau absolut ke file dataset MovieLens
ratings_path = 'ml-100k/u.data'
movies_path = 'ml-100k/u.item'

try:
    # Muat data ratings
    ratings = pd.read_csv(ratings_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

    # Muat data film
    movies = pd.read_csv(movies_path, sep='|', names=['movie_id', 'title'], usecols=[0, 1], encoding='latin-1')
except FileNotFoundError:
    app.logger.error(f"File tidak ditemukan. Pastikan path '{ratings_path}' dan '{movies_path}' benar.")
    exit()

# Gabungkan data rating dengan data film
data = pd.merge(ratings, movies, on='movie_id')

# Buat matriks user-item
user_movie_matrix = data.pivot_table(index='user_id', columns='title', values='rating')

# Menghitung similarity antar pengguna
user_movie_matrix_filled = user_movie_matrix.fillna(0)
user_similarity = cosine_similarity(user_movie_matrix_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Normalisasi judul film
movies['title'] = movies['title'].str.lower()
movie_titles = movies['title'].tolist()

# Fungsi rekomendasi berbasis kolaboratif dengan diversifikasi
def recommend_movies(user_id, num_recommendations=5):
    if user_id not in user_similarity_df.index:
        app.logger.debug(f"User ID {user_id} tidak ditemukan.")
        return []

    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    similar_users_ratings = user_movie_matrix.loc[similar_users]
    movie_list = similar_users_ratings.mean().sort_values(ascending=False)
    watched_movies = user_movie_matrix.loc[user_id].dropna().index
    recommendations = movie_list.drop(watched_movies)

    # Tambahkan diversifikasi dengan sedikit randomness
    recommendations = recommendations.sample(frac=1.0, random_state=42).head(num_recommendations)
    app.logger.debug(f"Recommendations for user {user_id}: {recommendations.index.tolist()}")
    return recommendations.index.tolist()

# Ekstraksi fitur konten
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['title'])

# Menghitung similarity antar film
movie_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movies['title'], columns=movies['title'])

# Fungsi rekomendasi berbasis konten
def recommend_similar_movies(movie_title, num_recommendations=5):
    movie_title = movie_title.lower()  # Normalisasi input judul film
    if movie_title not in movie_similarity_df.index:
        app.logger.debug(f"Movie title '{movie_title}' tidak ditemukan.")
        return []

    similar_movies = movie_similarity_df[movie_title].sort_values(ascending=False).index[1:num_recommendations+1]
    app.logger.debug(f"Similar movies for '{movie_title}': {similar_movies.tolist()}")
    return similar_movies.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form.get('user_id'))
    movie_title = request.form.get('movie_title')

    app.logger.debug(f"Received user_id: {user_id}, movie_title: {movie_title}")

    user_recommendations = []
    movie_recommendations = []

    # Validasi user_id
    if user_id in user_similarity_df.index:
        user_recommendations = recommend_movies(user_id)
    else:
        app.logger.debug(f"User ID {user_id} tidak ditemukan.")

    # Validasi movie_title
    if movie_title.lower() in movie_titles:
        movie_recommendations = recommend_similar_movies(movie_title)
    else:
        app.logger.debug(f"Movie title '{movie_title}' tidak ditemukan.")

    return render_template('index.html', user_recommendations=user_recommendations, movie_recommendations=movie_recommendations)

if __name__ == '__main__':
    # Memulai ngrok untuk membuat tunnel ke port lokal Flask (misalnya 5000)
    public_url = ngrok.connect(5000).public_url
    app.logger.info(f" * Running on {public_url}")
    app.run(debug=True)
