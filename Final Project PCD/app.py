from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
import joblib

from segmentation_pipeline import process_single_image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["RESULT_FOLDER"] = "static/results"

# ============================================
# LOAD MODEL + SCALER
# ============================================
bundle = joblib.load("model.pkl")
svm_model = bundle["svm_model"]
scaler = bundle["scaler"]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prob_normal = None
    prob_tbc = None
    filename = None
    mask_filename = None
    overlay_filename = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # ========================================================
            # FULL PIPELINE
            # ========================================================
            img1, mask_final, roi, feat, overlay = process_single_image(filepath)

            # Save mask
            mask_filename = "mask_" + filename
            cv2.imwrite(os.path.join(app.config["RESULT_FOLDER"], mask_filename), mask_final)

            # Save overlay
            overlay_filename = "overlay_" + filename
            cv2.imwrite(os.path.join(app.config["RESULT_FOLDER"], overlay_filename), overlay)

            # Predict
            feat_scaled = scaler.transform([feat])
            proba = svm_model.predict_proba(feat_scaled)[0]

            prob_normal = float(proba[0])
            prob_tbc = float(proba[1])

            pred = svm_model.predict(feat_scaled)[0]
            prediction = "TBC" if pred == 1 else "Normal"

    return render_template(
        "index.html",
        filename=filename,
        mask_filename=mask_filename,
        overlay_filename=overlay_filename,
        prediction=prediction,
        prob_normal=prob_normal,
        prob_tbc=prob_tbc
    )


if __name__ == "__main__":
    app.run(debug=True)
