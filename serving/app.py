"""
serving/app.py — Streamlit demo, now a thin UI layer over search_core.

Same UX as before (YOLO crop -> confirm/re-crop -> CLIP+BLIP search ->
caption re-rank), but all the actual model/index/champion logic now lives
in serving/search_core.py, shared with serving/api.py (FastAPI). This file
should only ever contain Streamlit widgets + calls into SearchEngine.

Run:
  cd MLOPS && streamlit run serving/app.py
"""

import os
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.search_core import SearchEngine  # noqa: E402

st.set_page_config(layout="wide", page_title="Visual Product Search", page_icon="👗")


@st.cache_resource
def load_engine():
    return SearchEngine()


clothing_choice = st.selectbox("Clothing Type", ["All", "Top", "Bottom", "Full Body"])
TYPE_MAP = {"All": None, "Top": 1, "Bottom": 2, "Full Body": 3}
requested_type = TYPE_MAP[clothing_choice]

st.title("👗 Visual Product Search Engine")
st.markdown("Upload a clothing image — the system will find visually and semantically similar products.")

with st.spinner("Loading models into memory..."):
    engine = load_engine()

st.caption(
    f"Serving champion config: `{engine.champion}`  "
    "(resolved from the MLflow Model Registry's Production stage, "
    "falling back to params.yaml → regression_gate.baseline_config if the registry is unreachable)"
)

uploaded_file = st.file_uploader("Upload a clothing image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_img = Image.open(uploaded_file).convert("RGB")
    st.markdown("---")
    st.subheader("Step 1: YOLO Product Detection")
    cropped_img, was_cropped, yolo_bbox = engine.crop_with_yolo(original_img, requested_type)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption="Original Image", use_container_width=True)
    with col2:
        if was_cropped:
            st.image(cropped_img, caption="YOLO Cropped Region", use_container_width=True)
        else:
            st.image(original_img, caption="No confident crop found", use_container_width=True)
            st.warning("YOLO didn't find a confident bounding box. Proceeding with the original image.")

    K = st.slider("Number of results to retrieve (K)", min_value=3, max_value=15, value=5)

    if st.button("🔍 Search Similar Products", type="primary", use_container_width=True):
        with st.spinner("Embedding query and searching HNSW index..."):
            results, meta = engine.search(original_img, k=K, requested_type=requested_type)

        st.markdown("---")
        if meta["query_caption"]:
            st.caption(f"BLIP caption: {meta['query_caption']}")
        st.subheader(f"Step 2: Top {len(results)} Matches")

        if not results:
            st.error("No results found for this query and filter combination. "
                      "(This is exactly what src/monitoring/canary_check.py watches for.)")

        cols_per_row = 5
        for row_idx in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, r in enumerate(results[row_idx: row_idx + cols_per_row]):
                with cols[col_idx]:
                    st.markdown(f"**Rank #{row_idx + col_idx + 1}**")
                    if os.path.exists(r["image_path"]):
                        st.image(Image.open(r["image_path"]), use_container_width=True)
                    else:
                        st.error("Image missing locally")
                    st.markdown(f"**Similarity:** {r['similarity']:.4f}")
                    st.caption(f"ID: `{r['item_id']}`  \n{r.get('caption', '')[:65]}")
