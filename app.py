import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the dataset
dff = pd.read_csv('books_data.csv')

# Combine relevant text columns to create a "content" field
dff['content'] = (
    dff['title'] + " " +
    dff['authors'] + " " 
    
)

# Load the SentenceTransformer model
@st.cache_resource
def load_model_and_embeddings(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data['content'].tolist())
    return model, embeddings

model, embeddings = load_model_and_embeddings(dff)
cosine_sim = cosine_similarity(embeddings)

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in dff['title'].values:
        return "Book not found in the dataset."
    
    # Find the index of the input title
    idx = dff[dff['title'] == title].index[0]
    
    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 recommendations
    book_indices = [i[0] for i in sim_scores]
    
    # Return recommended books (title, authors, and category)
    return dff[['title', 'authors']].iloc[book_indices]

# Streamlit App
st.title("Book Recommendation System")

input_title = st.text_input("Enter a book title:")

if st.button("Get Recommendations"):
    if input_title.strip():
        recommendations = get_recommendations(input_title)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.write("Top 5 Recommendations:")
            for _, row in recommendations.iterrows():
                st.write(f"**{row['title']}** by {row['authors']}")
    else:
        st.error("Please enter a valid book title.")
