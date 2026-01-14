from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        radius = float(request.form["radius"])
        mass = float(request.form["mass"])

        if 0.5 <= radius <= 2 and 0.1 <= mass <= 10:
            result = "Potentially Habitable 🌍"
        else:
            result = "Not Habitable ❌"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run()
