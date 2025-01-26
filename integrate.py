from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os

# Initialize Flask app
app = Flask(__name__)

# Set the GROQ_API_KEY environment variable
os.environ["GROQ_API_KEY"] = "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"

# Initialize Groq client
client = Groq()

# Load models and datasets
model = SentenceTransformer('all-MiniLM-L6-v2')
data = pd.read_csv('processed_dataset.csv')
embeddings = np.load('question_embeddings.npy')

# Crop recommendation model and scaler
crop_model = joblib.load('crop_recommendation_model.h5')
scaler = joblib.load('scaler.h5')

# Fertilizer recommendation model
def load_fertilizer_model():
    try:
        rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    return rf_model

rf_model = load_fertilizer_model()

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Fertilizer dictionary
fertilizer_dict = {
    0: 'Urea',
    1: 'DAP',
    2: 'Fourteen-Thirty Five-Fourteen',
    3: 'Twenty Eight-Twenty Eight',
    4: 'Seventeen-Seventeen-Seventeen',
    5: 'Twenty-Twenty',
    6: 'Ten-Twenty Six-Twenty Six'
}

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



@app.route('/soil_testing', methods=['GET', 'POST'])
def soil_testing():
    # Render the soil testing form page
    return render_template('soil_testing.html')



@app.route('/organic_farming', methods=['GET', 'POST'])
def organic_farming():
    # Render the organic farming page
    return render_template('organic_farming.html')

@app.route('/farmer_schemes', methods=['GET', 'POST'])
def farmer_schemes():
    # Render the farmer schemes page
    return render_template('farmer_schemes.html')

# Crop recommendation route
@app.route('/crop', methods=['GET', 'POST'])
def crop():
    if request.method == 'POST':
        try:
            # Extracting input data from the form
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Creating the feature array
            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

            # Scaling the features using the scaler
            scaled_features = scaler.transform(features)

            # Predicting the crop using the trained model
            prediction = crop_model.predict(scaled_features)
            crop_id = prediction[0]

            # Fetching the recommended crop from the dictionary
            recommended_crop = crop_dict.get(crop_id, "Unknown Crop")

            # Render the crop result page and display the recommended crop
            return render_template('crop.html', recommended_crop=recommended_crop)
        except Exception as e:
            # In case of any error, display the error message
            return render_template('crop.html', error=str(e))

    # Render the crop recommendation form page if it is a GET request
    return render_template('crop.html')

# Fertilizer recommendation route
@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():
    if request.method == 'POST':
        try:
            # Extracting input data from the form
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])

            # Preprocess the data and predict the fertilizer
            fertilizer_features = np.array([[N, P, K]])
            fertilizer_prediction = rf_model.predict(fertilizer_features)[0]  # Make prediction

            # Get the fertilizer name from the dictionary based on the prediction
            fertilizer_name = fertilizer_dict.get(fertilizer_prediction, "Unknown Fertilizer")

            # Render the fertilizer HTML page and display the result
            return render_template('fertilizer.html', fertilizer_prediction=fertilizer_name)
        except Exception as e:
            # In case of any error, display the error message on the page
            return render_template('fertilizer.html', error=str(e))
    
    # Render the fertilizer form page if it is a GET request
    return render_template('fertilizer.html')

# Chatbot route
@app.route('/chat', methods=['GET'])
def chatbot():
    return render_template('chatbot.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    if user_query.lower() in ["hi", "hello", "hai", "hii"]:
        return jsonify({'query': user_query, 'response': "Hello! Welcome to the Farmer's Chatbot. How can I assist you today?"})

    # Retrieve top answers
    top_answers = get_top_n_relevant_answers(user_query, data, embeddings, model)

    # Refine answer using Groq API
    refined_answer = get_refined_answer_from_groq(user_query, top_answers)

    return jsonify({'query': user_query, 'response': refined_answer})

# Home route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)









