import os
import joblib
import traceback
import numpy as np
import gradio as gr
import tensorflow as tf

# ==========================================================
# Load the Scaler and TensorFlow Model
# ==========================================================
try:
    scaler = joblib.load('breast_cancer_model.pkl')
    deployed_nn = tf.keras.models.load_model('breast_cancer_model.h5')
    print("Scaler and Deep Learning Model loaded successfully!")
except Exception as e:
    print(f"Warning: Files not found or error loading. {e}")
    scaler = None
    deployed_nn = None

# ==========================================================
# Prediction Function
# ==========================================================
def predict_cancer(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10):

    # User-provided Mean features
    user_mean_features = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

    # Predefined Error features
    preassumed_error_features = [
        0.2204,
        0.8561,
        1.778,
        16.64,
        0.005080,
        0.01104,
        0.0,
        0.0,
        0.01344,
        0.001784
    ]

    # Predefined Worst features
    preassumed_worst_features = [
        12.36,
        17.70,
        101.7,
        284.4,
        0.1216,
        0.1486,
        0.0,
        0.0,
        0.2226,
        0.07427
    ]

    # Combine into full 30-feature input
    full_30_features = (
        user_mean_features
        + preassumed_error_features
        + preassumed_worst_features
    )

    if deployed_nn is None or scaler is None:
        return "❌ Server Error: Model or Scaler failed to load."

    try:
        input_array = np.array([full_30_features])
        scaled_input = scaler.transform(input_array)

        prediction_prob = deployed_nn.predict(scaled_input)[0][0]

        if prediction_prob >= 0.5:
            return (
                f"🟢 Assessment Result (Confidence: {prediction_prob:.2%})\n\n"
                "Classification: BENIGN\n\n"
                "The cell characteristics suggest a non-cancerous tumor."
            )
        else:
            malignant_confidence = 1 - prediction_prob
            return (
                f"🔴 Assessment Result (Confidence: {malignant_confidence:.2%})\n\n"
                "Classification: MALIGNANT\n\n"
                "The cell characteristics indicate a high risk of cancer. "
                "Please consult an oncologist immediately."
            )

    except Exception:
        error_trace = traceback.format_exc()
        print(error_trace)
        return (
            "❌ Prediction failed due to an internal error.\n\n"
            f"{error_trace}"
        )

# ==========================================================
# Interface Setup
# ==========================================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate")) as app:

    gr.Markdown("<h1 style='text-align: center;'>🔬 Breast Cancer Detection System</h1>")
    gr.Markdown(
        "<p style='text-align: center;'>Adjust the medical metrics below. "
        "Advanced metrics are automatically estimated.</p>"
    )
    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            f1 = gr.Slider(0, 40, step=0.1, value=14.0, label="Mean Radius")
            f2 = gr.Slider(0, 50, step=0.1, value=19.0, label="Mean Texture")
            f3 = gr.Slider(0, 200, step=1.0, value=90.0, label="Mean Perimeter")
            f4 = gr.Slider(0, 3000, step=10.0, value=650.0, label="Mean Area")
            f5 = gr.Slider(0.0, 0.2, step=0.001, value=0.09, label="Mean Smoothness")

        with gr.Column():
            f6 = gr.Slider(0.0, 0.5, step=0.001, value=0.1, label="Mean Compactness")
            f7 = gr.Slider(0.0, 0.5, step=0.001, value=0.08, label="Mean Concavity")
            f8 = gr.Slider(0.0, 0.25, step=0.001, value=0.04, label="Mean Concave Points")
            f9 = gr.Slider(0.0, 0.5, step=0.001, value=0.18, label="Mean Symmetry")
            f10 = gr.Slider(0.0, 0.15, step=0.001, value=0.06, label="Mean Fractal Dimension")

    gr.Markdown("---")

    with gr.Row():
        submit_btn = gr.Button(
            "Run Neural Network Analysis",
            variant="primary",
            size="lg"
        )
        clear_btn = gr.ClearButton(size="lg")

    with gr.Row():
        result_box = gr.Textbox(
            label="Assessment Result",
            lines=5,
            interactive=False
        )

    gr.Markdown("""
---
### Breast Cancer Detection System

This application uses a trained deep learning model to predict whether the input characteristics are more consistent with a benign or malignant tumor.

**Disclaimer:** This application is intended for educational and research purposes only and should not be used as a substitute for professional medical advice or diagnosis.
""")

    input_components = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

    submit_btn.click(
        fn=predict_cancer,
        inputs=input_components,
        outputs=result_box
    )

    clear_btn.add(input_components + [result_box])

# ==========================================================
# Launch
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
    )