from transformers import AutoConfig, AutoModelForSequenceClassification

def create_model(config, num_labels):
    if config.model == "roberta-base":
        model_config = AutoConfig.from_pretrained(config.model)
        model_config.num_labels = num_labels
        model = AutoModelForSequenceClassification.from_pretrained(config.model, config=model_config)
    else:
        raise ValueError(f"Unsupported model: {config.model}")
    
    return model
