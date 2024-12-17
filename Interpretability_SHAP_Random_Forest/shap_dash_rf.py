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
import warnings

warnings.filterwarnings("ignore")

# Set the matplotlib backend to avoid GUI-related issues
matplotlib.use("Agg")

# Initialize Dash app and Flask caching
app = dash.Dash(__name__)
app.title = "Random Forest SHAP Dashboard"
cache = Cache(app.server, config={"CACHE_TYPE": "filesystem", "CACHE_DIR": "cache_dir"})

# Load the trained Random Forest model and dataset
best_rf_model = joblib.load("../Models/fraud_detection_random_forest_model.pkl")  # RF model path
X_test = pickle.load(open("../dataset/splits_pkl/X_test.pkl", "rb"))

# Handle NaN or Inf values by replacing them with zeros
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(0)

# Create SHAP explainer and calculate SHAP values for Random Forest
explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_test)

# Create necessary directories
os.makedirs("shap_visualizations", exist_ok=True)
os.makedirs("shap_visualizations/dependency_plots", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# For binary classifiers, handle scalar expected_value
if isinstance(explainer.expected_value, list) or isinstance(
    explainer.expected_value, np.ndarray
):
    expected_value = explainer.expected_value[1]  # Positive class index
else:
    expected_value = explainer.expected_value

# Generate and save force plot as an HTML file
shap.save_html(
    "assets/force_plot_rf.html",  # Save as RF-specific asset
    shap.force_plot(expected_value, shap_values, X_test.iloc[0, :]),
)

# Cache the dependence plot generation to avoid recalculating every time
@cache.memoize(timeout=60)  # Cache for 60 seconds
def generate_dependence_plot(feature):
    fig = plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature, shap_values[1], X_test, show=False)  # Use positive class
    plot_path = os.path.join("shap_visualizations/dependency_plots", f"dependence_plot_rf_{feature}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # Save plot as high-resolution image
    plt.close()
    return plot_path


# Function to encode the image to base64 for embedding in Dash
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded_image}"


# Dash app layout
app.layout = html.Div(
    style={
        "backgroundColor": "#f7f7f7",
        "padding": "20px",
    },
    children=[
        html.Div(
            style={
                "backgroundColor": "#34495e",
                "color": "#ecf0f1",
                "textAlign": "center",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.2)",
                "marginBottom": "20px",
            },
            children=[
                html.H1(
                    "Fraud Detection - Random Forest SHAP Dashboard",
                    style={
                        "fontSize": "32px",
                        "fontFamily": "Arial, sans-serif",
                        "margin": "0",
                    },
                )
            ],
        ),
        html.Div(
            style={
                "maxWidth": "1200px",
                "margin": "0 auto",
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
            },
            children=[
                html.Div(
                    style={
                        "backgroundColor": "#1abc9c",
                        "color": "#ffffff",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "marginBottom": "20px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.2)",
                    },
                    children=html.H2(
                        "Summary Plot",
                        style={
                            "fontSize": "24px",
                            "fontWeight": "bold",
                            "margin": "0",
                            "textAlign": "center",
                        },
                    ),
                ),
                html.Img(
                    src=encode_image("shap_visualizations/summary_plot_rf.png"),
                    style={
                        "width": "100%",
                        "maxWidth": "900px",
                        "height": "auto",
                        "marginBottom": "20px",
                        "borderRadius": "8px",
                    },
                ),
                html.Div(
                    style={
                        "backgroundColor": "#3498db",
                        "color": "#ffffff",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "marginBottom": "20px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.2)",
                    },
                    children=html.H2(
                        "Feature Importance Plot",
                        style={
                            "fontSize": "24px",
                            "fontWeight": "bold",
                            "margin": "0",
                            "textAlign": "center",
                        },
                    ),
                ),
                html.Img(
                    src=encode_image("shap_visualizations/feature_importance_rf.png"),
                    style={
                        "width": "100%",
                        "maxWidth": "900px",
                        "height": "auto",
                        "marginBottom": "20px",
                        "borderRadius": "8px",
                    },
                ),
                html.Div(
                    style={
                        "backgroundColor": "#9b59b6",
                        "color": "#ffffff",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "marginBottom": "20px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.2)",
                    },
                    children=html.H2(
                        "Force Plot",
                        style={
                            "fontSize": "24px",
                            "fontWeight": "bold",
                            "margin": "0",
                            "textAlign": "center",
                        },
                    ),
                ),
                html.Iframe(
                    src="assets/force_plot_rf.html",
                    style={
                        "width": "100%",
                        "height": "600px",
                        "border": "1px solid #bdc3c7",
                        "borderRadius": "8px",
                        "marginBottom": "20px",
                    },
                ),
                html.Div(
                    style={
                        "backgroundColor": "#e67e22",
                        "color": "#ffffff",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "marginBottom": "20px",
                        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.2)",
                    },
                    children=html.H2(
                        "Dependence Plot",
                        style={
                            "fontSize": "24px",
                            "fontWeight": "bold",
                            "margin": "0",
                            "textAlign": "center",
                        },
                    ),
                ),
                dcc.Dropdown(
                    id="feature-dropdown",
                    options=[{"label": feature, "value": feature} for feature in X_test.columns],
                    value=X_test.columns[0],
                    style={
                        "width": "50%",
                        "height": "40px",
                        "marginBottom": "20px",
                        "marginTop": "10px",
                        "borderRadius": "5px",
                        "fontSize": "16px",
                        "paddingLeft": "10px",
                    },
                ),
                html.Img(
                    id="dependence-plot",
                    style={
                        "width": "100%",
                        "maxWidth": "900px",
                        "height": "auto",
                        "borderRadius": "8px",
                    },
                ),
            ],
        ),
    ],
)


# Callback to update the dependence plot based on selected feature
@app.callback(
    dash.dependencies.Output("dependence-plot", "src"),
    [dash.dependencies.Input("feature-dropdown", "value")],
)
def update_dependence_plot(feature):
    plot_path = generate_dependence_plot(feature)
    return encode_image(plot_path)


if __name__ == "__main__":
    # Ensure the assets directory exists for saving plots
    os.makedirs("assets", exist_ok=True)
    app.run_server(debug=True)
