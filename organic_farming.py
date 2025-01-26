from flask import Flask, render_template

app = Flask(__name__)

@app.route('/organic_farming', methods=['GET', 'POST'])
def organic_farming():
    # Render the organic farming page
    return render_template('organic_farming.html')

if __name__ == '__main__':
    app.run(debug=True)
