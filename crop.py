from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the model and scaler
model = joblib.load('crop_recommendation_model.h5')
scaler = joblib.load('scaler.h5')

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Route to render the HTML form
@app.route('/')
def index():
    return render_template('crop.html')

# API route to handle crop recommendation
@app.route('/recommend', methods=['POST'])

def recommend():
    try:
        # Get input data from the form
        data = request.get_json()
        features = np.array([[data['N'], data['P'], data['K'], data['temperature'],
                              data['humidity'], data['ph'], data['rainfall']]])
        
        # Scale the input features
        scaled_features = scaler.transform(features)
        
        # Predict using the model
        prediction = model.predict(scaled_features)
        crop_id = prediction[0]
        
        # Map crop ID to crop name
        recommended_crop = crop_dict.get(crop_id, "Unknown Crop")
        
        return jsonify({"recommended_crop": recommended_crop})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)