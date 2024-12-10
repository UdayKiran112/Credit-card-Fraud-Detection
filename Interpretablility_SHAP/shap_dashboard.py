import dash
from dash import dcc, html
import base64
import os

def encode_image(image_path):
    """
    Encodes an image file to base64 for rendering in Dash.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded image string.
    """
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_image}"

# Define the Dash app
app = dash.Dash(__name__)
app.title = "SHAP Explainability Dashboard"

# Paths to SHAP visualizations
summary_plot_path = "shap_visualizations/summary_plot.png"
importance_plot_path = "shap_visualizations/feature_importance.png"
force_plot_path = "shap_visualizations/force_plot.html"
dependence_plot_path = "shap_visualizations/dependence_plot_V1.png"

# Load the force plot HTML
force_plot_html_path = "shap_visualizations/force_plot.html"
with open(force_plot_html_path, "rb") as f:
    force_plot_html = f.read().decode('utf-8', 'ignore')

# Define the app layout with better styling
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(
                    children="Fraud Detection - SHAP Explainability Dashboard",
                    style={
                        'textAlign': 'center',
                        'color': '#4CAF50',
                        'fontSize': '36px',
                        'fontFamily': 'Arial, sans-serif',
                        'marginTop': '20px'
                    }
                ),
                html.Div(
                    children=[
                        # Summary Plot
                        html.Div(
                            children=[
                                html.H2(
                                    children="Summary Plot",
                                    style={'textAlign': 'center', 'color': '#333', 'fontFamily': 'Arial, sans-serif'}
                                ),
                                html.Img(
                                    src=encode_image(summary_plot_path),
                                    style={'width': '100%', 'maxWidth': '900px', 'height': 'auto', 'marginBottom': '20px'}
                                ),
                            ],
                            style={'padding': '10px', 'backgroundColor': '#f4f4f9', 'borderRadius': '8px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'margin': '10px'}
                        ),
                        
                        # Feature Importance Plot
                        html.Div(
                            children=[
                                html.H2(
                                    children="Feature Importance Plot",
                                    style={'textAlign': 'center', 'color': '#333', 'fontFamily': 'Arial, sans-serif'}
                                ),
                                html.Img(
                                    src=encode_image(importance_plot_path),
                                    style={'width': '100%', 'maxWidth': '900px', 'height': 'auto', 'marginBottom': '20px'}
                                ),
                            ],
                            style={'padding': '10px', 'backgroundColor': '#f4f4f9', 'borderRadius': '8px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'margin': '10px'}
                        ),
                        
                        # Force Plot
                        html.Div(
                            children=[
                                html.H2(
                                    children="Force Plot",
                                    style={'textAlign': 'center', 'color': '#333', 'fontFamily': 'Arial, sans-serif'}
                                ),
                                html.Div(
                                    children=[
                                        html.Iframe(
                                            srcDoc=force_plot_html,
                                            style={'width': '100%', 'height': '600px', 'border': 'none', 'borderRadius': '8px'}
                                        ),
                                    ],
                                    style={'padding': '10px', 'backgroundColor': '#f4f4f9', 'borderRadius': '8px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'margin': '10px'}
                                ),
                            ]
                        ),
                        
                        # Dependence Plot
                        html.Div(
                            children=[
                                html.H2(
                                    children="Dependence Plot",
                                    style={'textAlign': 'center', 'color': '#333', 'fontFamily': 'Arial, sans-serif'}
                                ),
                                html.Img(
                                    src=encode_image(dependence_plot_path),
                                    style={'width': '100%', 'maxWidth': '900px', 'height': 'auto', 'marginBottom': '20px'}
                                ),
                            ],
                            style={'padding': '10px', 'backgroundColor': '#f4f4f9', 'borderRadius': '8px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'margin': '10px'}
                        ),
                    ],
                    style={'marginTop': '50px', 'padding': '20px'}
                ),
            ]
        ),
    ],
    style={
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': '#e8f4f8',
        'minHeight': '100vh',
        'padding': '20px'
    }
)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
