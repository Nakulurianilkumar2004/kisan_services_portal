from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os

app = Flask(__name__)

# Set the GROQ_API_KEY environment variable
os.environ["GROQ_API_KEY"] = "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"

# Initialize Groq client
client = Groq()

# Load model, dataset, and embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
data = pd.read_csv('processed_dataset.csv')
embeddings = np.load('question_embeddings.npy')

# Function to retrieve top N relevant answers
def get_top_n_relevant_answers(query, dataset, embeddings, model, top_n=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    top_answers = dataset.iloc[top_indices]['answers'].tolist()
    return top_answers

# Function to refine the answer using Groq API
def get_refined_answer_from_groq(query, top_answers):
    messages = [
        {
            "role": "user",
            "content": f"This is the user query: {query}\n\n"
                       f"Here are the top matched results from my knowledge base:\n"
                       + "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(top_answers)]) +
                       "\n\nFormulate a final answer most relevant to the user query."
        },
        {
            "role": "assistant",
            "content": "The final answer is:"
        }
    ]

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        top_p=1
    )
    return completion.choices[0].message.content
@app.route('/')
def home():
    return render_template('chatbot.html', chat_history=chat_history)

@app.route('/query', methods=['POST'])
def query():
    global chat_history
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    # Add the user query to chat history
    chat_history.append({'sender': 'user', 'text': user_query})

    if user_query.lower() in ["hi", "hello", "hai", "hii"]:
        response_text = "Hello! Welcome to the Farmer's Chatbot. How can I assist you today?"
        chat_history.append({'sender': 'bot', 'text': response_text})
        return jsonify({'query': user_query, 'response': response_text})

    # Retrieve top answers and refine response
    top_answers = get_top_n_relevant_answers(user_query, data, embeddings, model)
    refined_answer = get_refined_answer_from_groq(user_query, top_answers)

    # Add the bot's response to chat history
    chat_history.append({'sender': 'bot', 'text': refined_answer})

    return jsonify({'query': user_query, 'response': refined_answer})

if __name__ == '__main__':
    app.run(debug=True)