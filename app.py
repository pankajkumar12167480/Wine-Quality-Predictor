from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open('models/rf.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Extract form inputs
            fixed_acidity = float(request.form["fixed_acidity"])
            volatile_acidity = float(request.form["volatile_acidity"])
            citric_acid = float(request.form["citric_acid"])
            residual_sugar = float(request.form["residual_sugar"])
            chlorides = float(request.form["chlorides"])
            free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
            total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
            density = float(request.form["density"])
            ph = float(request.form["ph"])
            sulphates = float(request.form["sulphates"])
            alcohol = float(request.form["alcohol"])

            # Make array for prediction
            features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                  chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                                  density, ph, sulphates, alcohol]])

            prediction = model.predict(features)[0]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
