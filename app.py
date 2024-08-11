import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
@flask_app.route("/")
def Home():
    return render_template("index.html")
@flask_app.route("/service")
def service():
    return render_template("matrix_services.html")

@flask_app.route("/about")
def about():
    return render_template("matrix_about_us.html")

@flask_app.route("/contact")
def contact():
    return render_template("contact.html")
@flask_app.route("/predict", methods = ["POST"])
def predict():
    df = pd.read_csv('collegePlace (1).csv')
    df['Gender'] = df['Gender'].map({
        'Male': 1,
        'Female': 0})
    df['Stream'] = df['Stream'].map({
        'Electronics And Communication': 1,
        'Computer Science': 2,
        'Information Technology': 3,
        'Mechanical': 4,
        'Electrical': 5,
        'Civil': 6})

    print(df.head())
    X = df.drop(columns=['PlacedOrNot'], axis=1)
    y = df['PlacedOrNot']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)

    clf = RandomForestClassifier()

    # Fit the model
    clf.fit(X_train, y_train)
    age = 22
    gender = 1  # 1=Male, 0=Female
    stream =3# Electronics And Communication': 1,
    #              'Computer Science': 2,
    #              'Information Technology': 3,
    #              'Mechanical':4,
    #              'Electrical':5,
    #              'Civil':6
    Internships = 4
    CGPA = 9
    Hostel = 1  # 1= stay in hostel, 0=not staying in hostel
    HistoryOfBacklogs = 0# 1 = had backlogs, 0=no backlogs
    prediction = clf.predict([[age, gender, stream, Internships, CGPA, Hostel, HistoryOfBacklogs]])
    return render_template("matrix_services.html", prediction_text = "PLACEMENT RESULT: {}".format(prediction))



if __name__ == "__main__":
    flask_app.run(debug=True)