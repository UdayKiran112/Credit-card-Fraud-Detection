import dash
from dash import dcc, html
import shap
import joblib
import pickle
import base64
import os
from flask_caching import Cache
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import plotly.graph_objs as go

# Set the matplotlib backend to avoid GUI-related issues
matplotlib.use('Agg')

# Initialize Dash app and Flask caching
app = dash.Dash(__name__)
app.title = "SHAP Explainability Dashboard"
cache = Cache(app.server, config={'CACHE_TYPE': 'filesystem', 'CACHE_DIR': 'cache_dir'})

# Load the trained model and dataset
best_model = joblib.load('../Models/fraud_detection_xgboost_model.pkl')
X_test = pickle.load(open('../dataset/splits_pkl/X_test.pkl', 'rb'))

# Handle NaN or Inf values by replacing them with zeros (or use another imputation strategy)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(0)

# Create SHAP explainer and calculate SHAP values
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Cache the dependence plot generation to avoid recalculating every time
@cache.memoize(timeout=60)  # Cache for 60 seconds
def generate_dependence_plot(feature):
    fig = plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature, shap_values, X_test, show=False)
    plot_path = "shap_visualizations/dependence_plot.png"
    plt.savefig(plot_path, dpi=300)  # Save the plot as a high-resolution image
    plt.close()
    return plot_path

# Function to encode the image to base64 for embedding in Dash
def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_image}"

# Dash app layout
app.layout = html.Div(
    style={'backgroundColor': '#f7f7f7', 'padding': '20px'},  # Background color for the page
    children=[
        html.H1(
            "Fraud Detection - SHAP Explainability Dashboard",
            style={'textAlign': 'center', 'color': '#2C3E50', 'fontSize': '32px', 'fontFamily': 'Arial, sans-serif'}
        ),
        html.Div(
            style={'maxWidth': '1200px', 'margin': '0 auto', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'},
            children=[
                html.H2("Summary Plot", style={'color': '#2C3E50', 'fontSize': '24px', 'fontWeight': 'bold'}),
                html.Img(
                    src=encode_image("shap_visualizations/summary_plot.png"),
                    style={'width': '100%', 'maxWidth': '900px', 'height': 'auto', 'marginBottom': '20px', 'borderRadius': '8px'}
                ),
                
                html.H2("Feature Importance Plot", style={'color': '#2C3E50', 'fontSize': '24px', 'fontWeight': 'bold'}),
                html.Img(
                    src=encode_image("shap_visualizations/feature_importance.png"),
                    style={'width': '100%', 'maxWidth': '900px', 'height': 'auto', 'marginBottom': '20px', 'borderRadius': '8px'}
                ),
                
                html.H2("Dependence Plot", style={'color': '#2C3E50', 'fontSize': '24px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': feature, 'value': feature} for feature in X_test.columns],
                    value=X_test.columns[0],
                    style={
                        'width': '50%', 
                        'height': '40px',
                        'marginBottom': '20px', 
                        'marginTop': '20px', 
                        'borderRadius': '5px', 
                        'backgroundColor': '#ecf0f1',
                        'border': '1px solid #bdc3c7',
                        'fontSize': '16px',  # Added for better readability
                        'paddingLeft': '10px',
                    }
                ),
                html.Img(
                    id='dependence-plot',
                    style={'width': '100%', 'maxWidth': '900px', 'height': 'auto', 'borderRadius': '8px'}
                ),
            ]
        ),
    ]
)

# Callback to update the dependence plot based on selected feature
@app.callback(
    dash.dependencies.Output('dependence-plot', 'src'),
    [dash.dependencies.Input('feature-dropdown', 'value')]
)
def update_dependence_plot(feature):
    plot_path = generate_dependence_plot(feature)
    return encode_image(plot_path)

if __name__ == "__main__":
    app.run_server(debug=True)
