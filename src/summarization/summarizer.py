from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "t5-small"


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def summarize_article(article, max_input_length=512, max_summary_length=100, min_summary_length=20):
    """
    Generate a short summary for a news article.
    """

    input_text = "summarize: " + article

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_summary_length,
        min_length=min_summary_length,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary