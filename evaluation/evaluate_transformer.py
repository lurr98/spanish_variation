from transformers import AutoModelForSequenceClassification, BertTokenizer, Trainer, TrainingArguments


def load_fine_tuned_model(model_path: str) -> AutoModelForSequenceClassification:

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return model


# def evaluate_transformer_model(model: AutoModelForSequenceClassification) -> str:

    