import streamlit as st
import base64
import tempfile
import os
from inference_sdk import InferenceHTTPClient
from PIL import Image
import json

st.set_page_config(page_title="Dental Disease Detection", layout="wide")

st.title("🦷 Dental Disease Detection")
st.write("Upload a dental image to detect cavities, gingivitis, discoloration, and ulcers.")

# Sidebar controls
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0,
    1.0,
    0.3,
    0.05
)

# API key input
api_key = st.sidebar.text_input(
    "Roboflow API Key",
    type="password"
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose a dental image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and api_key:

    # Display input image
    col1, col2 = st.columns(2)

    with col1:
        st.header("Input Image")
        input_image = Image.open(uploaded_file)
        st.image(input_image, use_container_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp:

        input_image.save(tmp.name)
        tmp_path = tmp.name

    # Initialize Roboflow client
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )

    with st.spinner("Running dental disease detection..."):

        try:

            # Run workflow
            result = client.run_workflow(
                workspace_name="sagiru-garba",
                workflow_id="dental-disease-detection-1779011702066",
                images={"image": tmp_path},
                parameters={
                    "confidence": confidence_threshold
                },
                use_cache=True
            )

            # Get results
            item = result[0]

            # Display annotated image
            with col2:

                st.header("Detected Conditions")

                if "annotated_image" in item:

                    annotated_b64 = item["annotated_image"]

                    if isinstance(annotated_b64, dict):
                        annotated_b64 = (
                            annotated_b64.get("value")
                            or annotated_b64.get("base64")
                            or annotated_b64.get("data")
                        )

                    annotated_bytes = base64.b64decode(
                        annotated_b64
                    )

                    st.image(
                        annotated_bytes,
                        use_container_width=True
                    )

                    st.download_button(
                        label="Download Annotated Image",
                        data=annotated_bytes,
                        file_name="dental_detection_result.jpg",
                        mime="image/jpeg"
                    )

                else:
                    st.warning(
                        "No annotated image returned."
                    )

            # Show predictions table
            st.header("Detection Results")

            predictions = (
                item.get("predictions", {}).get(
                    "predictions",
                    []
                )
                if isinstance(
                    item.get("predictions"),
                    dict
                )
                else []
            )

            detection_count = item.get(
                "detection_count",
                len(predictions)
            )

            st.metric(
                "Total Detections",
                detection_count
            )

            if predictions:

                import pandas as pd

                pred_data = []

                for pred in predictions:

                    pred_data.append({
                        "Class": pred.get(
                            "class",
                            "unknown"
                        ),
                        "Confidence":
                        f"{pred.get('confidence', 0):.2%}",
                        "X": int(pred.get("x", 0)),
                        "Y": int(pred.get("y", 0)),
                        "Width": int(
                            pred.get("width", 0)
                        ),
                        "Height": int(
                            pred.get("height", 0)
                        )
                    })

                df = pd.DataFrame(pred_data)

                st.dataframe(
                    df,
                    use_container_width=True
                )

                # Class distribution
                st.subheader(
                    "Condition Distribution"
                )

                class_counts = (
                    df["Class"].value_counts()
                )

                st.bar_chart(class_counts)

            else:
                st.info(
                    "No dental conditions detected above "
                    "the confidence threshold."
                )

            # Show raw JSON
            with st.expander(
                "View Raw JSON"
            ):
                st.json(item)

        except Exception as e:

            st.error(
                f"Error running detection: {str(e)}"
            )

        finally:

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

elif uploaded_file is not None and not api_key:

    st.warning(
        "Please enter your Roboflow API Key "
        "in the sidebar."
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** gingivitis-n2cjt/3"
)
st.sidebar.markdown(
    "**Classes:** caries, gingivitis, "
    "tooth discoloration, ulcer"
)