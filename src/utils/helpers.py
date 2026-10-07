import torch

from src.preprocessing.cleaning import clean_text
from src.summarization.summarizer import summarize_article


LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}


def analyze_news(article, model, vocab, max_length=100):

    # Apply the same preprocessing used during model training
    cleaned_article = clean_text(article)

    # Convert text into word sequence
    words = cleaned_article.split()

    sequence = []

    for word in words:
        sequence.append(
            vocab.get(word, vocab["<unk>"])
        )

    # Truncate or pad sequence
    sequence = sequence[:max_length]

    if len(sequence) < max_length:
        sequence += [
            vocab["<pad>"]
        ] * (max_length - len(sequence))

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(
        [sequence],
        dtype=torch.long
    )

    # Classification
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0, predicted_class
        ].item()

    category = LABEL_NAMES[predicted_class]

    # Generate summary
    summary = summarize_article(article)

    return {
        "category": category,
        "confidence": confidence,
        "summary": summary
    }