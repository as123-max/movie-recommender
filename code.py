import pandas as pd
import streamlit as st

# Load the Excel file
df = pd.read_excel('movies.xlsx', header=1)

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Check columns
print(df.columns)

# Optional: Check dataset size and first rows
print(df.shape)
print(df['title'])
print(df.iloc[0])
df['content'] = df['genre'] + " " + df['overview']
df['content'] = df['content'].str.lower()
df['content'] = df['content'].str.strip()
print(df[['title', 'content']].head())

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])
print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)
print(cosine_sim.shape)

def recommend(title,num_recommendations=5):
    idx = df[df['title'].str.lower() == title.lower()].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()
print(recommend('Titanic'))

st.title('Movie Recommendation System')
st.write('Enter a movie title to get recommendations.')

movie_list = df['title'].tolist()

selected_movie = st.selectbox('Select movie title:', movie_list)
if st.button('Get Recommendations'):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.subheader('Recommended Movies:')
        for i,movie in enumerate(recommendations,1):
            st.write(f"{i}. {movie}")
print("everything ran successfully")
