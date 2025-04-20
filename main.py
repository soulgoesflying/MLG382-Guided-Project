import dash
import dash_auth
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import joblib

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
model = joblib.load('final_mlp_model.pkl')

# Layout
app.layout = html.Div(
    className='container',
    children=[
        html.H1("🎓 BrightPath Predictor", style={'textAlign': 'center'}),

        html.Label("Age:"),
        dcc.Input(id='age', type='number', value=16, min=15, max=18),

        html.Label("GPA:"),
        dcc.Input(id='gpa', type='number', value=3.0, step=0.1, min=0.0, max=4.0),

        html.Label("Weekly Study Time (hrs):"),
        dcc.Input(id='study_time', type='number', value=10, min=0, max=20),

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

        html.Button("Predict Grade", id='predict-btn', n_clicks=0),
        html.Div(id='prediction-output')
    ]
)

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('age', 'value'),
    State('gpa', 'value'),
    State('study_time', 'value'),
    State('parental_support', 'value'),
    State('music', 'value')
)
def predict_grade(n_clicks, age, gpa, study_time, parental_support, music):
    if n_clicks > 0:
        features = [[age, gpa, study_time, parental_support, music]]
        prediction = model.predict(features)[0]

        # Optional: map prediction to letter grade
        grade_map = {0: 'F', 1: 'D', 2: 'C', 3: 'B', 4: 'A'}
        grade_letter = grade_map.get(prediction, prediction)

        return f"🎓 Predicted Grade: {grade_letter}"
    return ""

if __name__ == '__main__':
    app.run(debug=True, port=8051)
