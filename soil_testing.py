from flask import Flask, render_template

app = Flask(__name__)

@app.route('/soil_testing', methods=['GET', 'POST'])
def soil_testing():
    # Render the soil testing form page
    return render_template('soil_testing.html')

if __name__ == '__main__':
    app.run(debug=True)


