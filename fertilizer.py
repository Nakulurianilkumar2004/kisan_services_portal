from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the fertilizer recommendation model
def load_model():
    try:
        rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    return rf_model

rf_model = load_model()

# Mapping fertilizer prediction to their respective names
fertilizer_dict = {
    0: 'Urea',
    1: 'DAP',
    2: 'Fourteen-Thirty Five-Fourteen',
    3: 'Twenty Eight-Twenty Eight',
    4: 'Seventeen-Seventeen-Seventeen',
    5: 'Twenty-Twenty',
    6: 'Ten-Twenty Six-Twenty Six'
}

@app.route('/fertilizer', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Extracting input data from the form
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])

            # Preprocess the data and predict
            fertilizer_features = np.array([[N, P, K]])
            fertilizer_prediction = rf_model.predict(fertilizer_features)[0]

            # Get the fertilizer name
            fertilizer_name = fertilizer_dict.get(fertilizer_prediction, "Unknown Fertilizer")

            return render_template('fertilizer.html', fertilizer_prediction=fertilizer_name)
        except Exception as e:
            return render_template('fertilizer', error=str(e))

    return render_template('fertilizer.html')

if __name__ == '__main__':
    app.run(debug=True)
    
    