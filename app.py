from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("calorie_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = int(request.form['age'])
        weight = int(request.form['weight'])
        duration = int(request.form['duration'])
        goal = request.form['goal']

        calories = model.predict([[age, weight, duration]])[0]

        if goal == "weight_loss":
            workout = "Pushups + Yoga"
            diet = "Low Carb, High Protein"
        elif goal == "muscle_gain":
            workout = "Strength Training"
            diet = "High Protein, High Calories"
        else:
            workout = "Mixed Workout"
            diet = "Balanced Diet"

        return render_template("index.html",
                               calories=round(calories,2),
                               workout=workout,
                               diet=diet)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
