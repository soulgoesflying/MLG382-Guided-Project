import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import joblib
import os
import datetime
from dash_auth import BasicAuth

# --- 1. SETUP & AUTH ---
VALID_USERNAME_PASSWORD_PAIRS = {'teacher': 'bright123', 'admin': 'adminpass'}
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP])
auth = BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)

# --- 2. LOAD ARTIFACTS ---
model = joblib.load('artifacts/mlp_model_smote.pkl')
scaler = joblib.load('artifacts/mlp_scaler.pkl')

# --- 3. UI COMPONENTS ---
def create_input_field(label, component):
    return html.Div([dbc.Label(label), component, html.Br()], className="mb-2")

app.layout = dbc.Container([
    dbc.Row([
    dbc.Col(html.H1([
        html.I(className="bi bi-mortarboard-fill me-3"), 
        "BRIGHTPATH PREDICTOR v2.0"
    ], className="text-center my-5 fw-bold glass-header"), width=12) # Added 'glass-header'
]),

    dbc.Row([
        # LEFT COLUMN
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Student Assessment Form", className="mb-0")),
                dbc.CardBody([
                    html.H6("Academic Performance", className="text-muted border-bottom mb-3"),
                    create_input_field("Weekly Study Time", dbc.Input(id='study_time', type='number', value=10)),
                    create_input_field("Recent Absences", dbc.Input(id='absences', type='number', value=0)),
                    create_input_field("Tutoring Participation", dbc.Select(id='tutoring', options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}], value=0)),
                    
                    html.H6("Demographics & Support", className="text-muted border-bottom mt-4 mb-3"),
                    dbc.Row([
                        dbc.Col(create_input_field("Age", dbc.Input(id='age', type='number', value=16))),
                        dbc.Col(create_input_field("Gender", dbc.Select(id='gender', options=[{"label": "Male", "value": 0}, {"label": "Female", "value": 1}], value=0))),
                    ]),
                    create_input_field("Ethnicity", dbc.Select(id='ethnicity', options=[{"label": e, "value": i} for i, e in enumerate(['Black','White','Hispanic','Asian','Other'])], value=0)),
                    create_input_field("Parental Education", dbc.Select(id='parent_ed', options=[{"label": e, "value": i} for i, e in enumerate(['None','High School','College','Graduate'])], value=1)),
                    create_input_field("Parental Support Level", dbc.Select(id='parent_support', options=[{"label": s, "value": i} for i, s in enumerate(['None','Low','Moderate','High','Very High'])], value=2)),

                    html.H6("Extracurriculars", className="text-muted border-bottom mt-4 mb-3"),
                    dbc.Checklist(
                        options=[
                            {"label": "Extracurricular Activities", "value": "extra"},
                            {"label": "Sports Participation", "value": "sports"},
                            {"label": "Music Programs", "value": "music"},
                            {"label": "Volunteering", "value": "volunteer"},
                        ],
                        value=[], id="activities-checklist", switch=True,inline=False,labelStyle={"display": "block", "marginBottom": "10px"}
                    ),
                    dbc.Button("GENERATE ANALYSIS", id='predict-btn', size="lg", className="w-100 mt-4")
                ])
            ])
        ], lg=5),

        # RIGHT COLUMN
        dbc.Col([
    # 1. LIVE DIAGNOSTIC CARD
    dbc.Card([
        # FIXED: Added missing comma and moved className inside CardHeader
        dbc.CardHeader(html.H5("Live Diagnostic Result", className="mb-0")),
        dbc.CardBody([
            html.Div(id='prediction-output', className="text-center mb-4"),
            dcc.Graph(id='risk-gauge', config={'displayModeBar': False}),
            html.Div(id='intervention-alert')
        ])
    ], className="mb-4"), # Spacing between cards

    # 2. SYSTEM LOGS CARD
    dbc.Card([
        # FIXED: Wrapped text in html.H5 so the CSS can find and glow it
        dbc.CardHeader(html.H5("System Logs", className="mb-0")),
        dbc.CardBody([
            html.Div(id='log-status'),
            dbc.Button("Download Intervention List", id="btn-download", color="info", size="sm", className="w-100 mt-2"),
            dcc.Download(id="download-intervention-csv")
        ])
    ])
], lg=7)

# --- 4. CALLBACK ---
@app.callback(
    [Output('prediction-output', 'children'),
     Output('risk-gauge', 'figure'),
     Output('intervention-alert', 'children'),
     Output('log-status', 'children')],
    Input('predict-btn', 'n_clicks'),
    [State('age', 'value'), State('gender', 'value'), State('ethnicity', 'value'),
     State('parent_ed', 'value'), State('study_time', 'value'), State('absences', 'value'),
     State('tutoring', 'value'), State('parent_support', 'value'),
     State('activities-checklist', 'value')]
)
def run_analysis(n_clicks, age, gender, ethnicity, ped, study, absce, tutor, psupp, activities):
    if not n_clicks:
        # Transparent Awaiting Data screen
        fig = go.Figure()
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          xaxis={'visible': False}, yaxis={'visible': False})
        return html.H3("AWAITING STUDENT DATA", className="pt-5"), fig, "", ""

    extra = 1 if "extra" in activities else 0
    sports = 1 if "sports" in activities else 0
    music = 1 if "music" in activities else 0
    volunteer = 1 if "volunteer" in activities else 0

    raw_features = [[float(age), int(gender), int(ethnicity), int(ped), float(study), 
                     float(absce), int(tutor), int(psupp), extra, sports, music, volunteer]]
    
    scaled = scaler.transform(raw_features)
    prediction = model.predict(scaled)[0]
    res_grade = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}.get(prediction, "N/A")
    risk_val = (prediction / 4) * 100

    # Neon Transparent Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk_val,
        number={'suffix': "%", 'font': {'size': 80, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "white"},
            'bar': {'color': "white"},
            'bgcolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 40], 'color': "#00ff88"}, # Neon Green
                {'range': [40, 70], 'color': "#ffcc00"}, # Neon Amber
                {'range': [70, 100], 'color': "#ff3366"} # Neon Red
            ]
        }
    ))
        # Force the graph to be invisible so the background shows through
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Inter"},
        height=380,
        margin=dict(l=30, r=30, t=80, b=20)
    )
    
    # Make the gauge segments vibrant neon
    fig.update_traces(
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "white", 'tickwidth': 2},
            'bar': {'color': "white", 'thickness': 0.25},
            'bgcolor': "rgba(255,255,255,0.05)",
            'steps': [
                {'range': [0, 40], 'color': "#00ff88"}, # Neon Green
                {'range': [40, 70], 'color': "#ffcc00"}, # Neon Gold
                {'range': [70, 100], 'color': "#ff3366"} # Neon Red
            ]
        },
        number={'font': {'size': 90, 'color': 'white'}, 'suffix': "%"}
    )


    # Logging Logic
    alert = ""
    log_msg = dbc.Badge("Record Optimized", color="success")
    if res_grade in ['D', 'F']:
        alert = dbc.Alert(f"INTERVENTION TRIGGERED: Grade {res_grade}.", color="danger", className="mt-3")
        log_msg = dbc.Badge(f"Logged: Grade {res_grade}", color="warning")

    grade_display = html.H2(f"PREDICTED GRADE: {res_grade}", 
                            style={'color': '#ff3366' if res_grade in ['D','F'] else '#00ff88', 
                                   'fontWeight': '900', 'textShadow': '0 0 15px currentColor'})

    return grade_display, fig, alert, log_msg

@app.callback(
    Output("download-intervention-csv", "data"),
    Input("btn-download", "n_clicks"),
    prevent_initial_call=True,
)
def download_csv(n_clicks):
    if os.path.exists('intervention_list.csv'):
        return dcc.send_file('intervention_list.csv')
    return dash.no_update

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)