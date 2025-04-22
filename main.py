import dash
import dash_auth
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import joblib
from flask import Flask


# Basic login setup
VALID_USERNAME_PASSWORD_PAIRS = {
    'teacher': 'bright123',
    'admin': 'adminpass'
}

# Initialize Dash app
app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)
server = app.server
server.secret_key = os.environ.get('SECRET_KEY', 'brightpath_default_key')

# Load ML model
model = joblib.load('artifacts/mlp_model_smote.pkl')
scaler = joblib.load('artifacts/mlp_scaler.pkl')


# Layout
app.layout = html.Div(
    className='container',
    children=[
        html.H1("🎓 BrightPath Predictor", style={'textAlign': 'center'}),

        html.Label("Age:"),
        dcc.Input(id='age', type='number', value=16, min=15, max=18),

        html.Label("Gender:"),
        dcc.RadioItems(
            id='gender',
            options=[
                {'label': 'Male', 'value': 0},
                {'label': 'Female', 'value': 1},
                {'label': 'Other', 'value': 2}
            ],
            value=0,
            inline=True
        ),

        html.Label("Ethnicity:"),
        dcc.Dropdown(
            id='ethnicity',
            options=[
                {'label': 'Black', 'value': 0},
                {'label': 'White', 'value': 1},
                {'label': 'Hispanic', 'value': 2},
                {'label': 'Asian', 'value': 3},
                {'label': 'Other', 'value': 4}
            ],
            value=0
        ),

        html.Label("Parental Education Level:"),
        dcc.Dropdown(
            id='parental_education',
            options=[
                {'label': 'None', 'value': 0},
                {'label': 'High School', 'value': 1},
                {'label': 'College', 'value': 2},
                {'label': 'Graduate Degree', 'value': 3}
            ],
            value=1
        ),

        html.Label("Weekly Study Time (hrs):"),
        dcc.Input(id='study_time', type='number', value=10, min=0, max=40),

        html.Label("Absences:"),
        dcc.Input(id='absences', type='number', value=0, min=0),

        html.Label("Tutoring:"),
        dcc.RadioItems(
            id='tutoring',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0,
            inline=True
        ),

        html.Label("Parental Support:"),
        dcc.Dropdown(
            id='parental_support',
            options=[
                {'label': 'None', 'value': 0},
                {'label': 'Low', 'value': 1},
                {'label': 'Moderate', 'value': 2},
                {'label': 'High', 'value': 3},
                {'label': 'Very High', 'value': 4},
            ],
            value=2
        ),

        html.Label("Extracurricular Activities:"),
        dcc.RadioItems(
            id='extracurricular',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0,
            inline=True
        ),

        html.Label("Sports Participation:"),
        dcc.RadioItems(
            id='sports',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0,
            inline=True
        ),

        html.Label("Participates in Music?"),
        dcc.RadioItems(
            id='music',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0,
            inline=True
        ),

        html.Label("Volunteering:"),
        dcc.RadioItems(
            id='volunteering',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0,
            inline=True
        ),

        html.Br(),
        html.Button("Predict Grade", id='predict-btn', n_clicks=0),
        html.Div(id='prediction-output', style={'marginTop': '20px', 'fontWeight': 'bold'})
    ]
)

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('age', 'value'),
    State('gender', 'value'),
    State('ethnicity', 'value'),
    State('parental_education', 'value'),
    State('study_time', 'value'),
    State('absences', 'value'),
    State('tutoring', 'value'),
    State('parental_support', 'value'),
    State('extracurricular', 'value'),
    State('sports', 'value'),
    State('music', 'value'),
    State('volunteering', 'value')
)
def predict_grade(n_clicks, age, gender, ethnicity, parental_education, study_time,
                  absences, tutoring, parental_support, extracurricular, sports,
                  music, volunteering):
    if n_clicks > 0:
        features = [[
            age, gender, ethnicity, parental_education, study_time,
            absences, tutoring, parental_support, extracurricular,
            sports, music, volunteering
        ]]
        prediction = model.predict(features)[0]

        grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
        grade_letter = grade_map.get(prediction, prediction)

        if grade_letter == 'F':
            return "⚠️ You are an at-risk student. Predicted grade: F"
        else:
            return f"✅ Your predicted grade is: {grade_letter}"

    return ""


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)


