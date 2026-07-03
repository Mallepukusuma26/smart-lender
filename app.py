from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/result", methods=["POST"])
def result():

    try:
        Gender = float(request.form["Gender"])
        Married = float(request.form["Married"])
        Dependents = float(request.form["Dependents"])
        Education = float(request.form["Education"])
        Self_Employed = float(request.form["Self_Employed"])
        ApplicantIncome = float(request.form["ApplicantIncome"])
        CoapplicantIncome = float(request.form["CoapplicantIncome"])
        LoanAmount = float(request.form["LoanAmount"])
        Loan_Amount_Term = float(request.form["Loan_Amount_Term"])
        Credit_History = float(request.form["Credit_History"])
        Property_Area = float(request.form["Property_Area"])

        features = np.array([[
            Gender,
            Married,
            Dependents,
            Education,
            Self_Employed,
            ApplicantIncome,
            CoapplicantIncome,
            LoanAmount,
            Loan_Amount_Term,
            Credit_History,
            Property_Area
        ]])

        prediction = model.predict(features)

        if prediction[0] == 1:
            result_text = "✅ Loan Approved"
        else:
            result_text = "❌ Loan Rejected"

        return render_template(
            "result.html",
            prediction=result_text
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
