import sys
from pathlib import Path
import json

import streamlit as st
import torch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.pytorch.model import NewsClassifier
from src.utils.helpers import analyze_news


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="News Intelligence System",
    page_icon="📰",
    layout="wide"
)


# --------------------------------------------------
# Load Vocabulary
# --------------------------------------------------

@st.cache_resource
def load_vocabulary():

    vocab_path = PROJECT_ROOT / "data" / "processed" / "vocab.json"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    return vocab


# --------------------------------------------------
# Load PyTorch Model
# --------------------------------------------------

@st.cache_resource
def load_classification_model(vocab):

    model = NewsClassifier(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=128,
        num_classes=4,
        dropout=0.3
    )

    model_path = (
        PROJECT_ROOT
        / "models"
        / "pytorch"
        / "best_news_classifier.pt"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        )
    )

    model.eval()

    return model


# --------------------------------------------------
# Load Resources
# --------------------------------------------------

vocab = load_vocabulary()
classification_model = load_classification_model(vocab)


# --------------------------------------------------
# Application Header
# --------------------------------------------------

st.title("📰 AI-Based News Intelligence System")

st.write(
    """
    This application uses deep learning to classify news articles
    into different topics and automatically generate a short summary.
    """
)

st.divider()


# --------------------------------------------------
# News Article Input
# --------------------------------------------------

st.subheader("Enter a News Article")

article = st.text_area(
    "Paste your news article below:",
    height=300,
    placeholder="Paste a news article here..."
)


# --------------------------------------------------
# Analyze Button
# --------------------------------------------------

if st.button(
    "🔍 Analyze Article",
    type="primary"
):

    if not article.strip():

        st.warning(
            "Please enter a news article first."
        )

    else:

        with st.spinner(
            "Analyzing article..."
        ):

            result = analyze_news(
                article,
                classification_model,
                vocab
            )

        st.success(
            "Analysis completed successfully!"
        )

        st.divider()

        # --------------------------------------------------
        # Classification Result
        # --------------------------------------------------

        st.subheader("📌 News Classification")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Topic",
                result["category"]
            )

        with col2:

            st.metric(
                "Confidence",
                f"{result['confidence']:.2%}"
            )

        st.divider()

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        st.subheader("📝 AI-Generated Summary")

        st.info(
            result["summary"]
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Powered by PyTorch, TensorFlow/Keras, Hugging Face Transformers and T5-small"
)