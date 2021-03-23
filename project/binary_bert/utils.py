import transformers
import torch


def get_model_and_tokenizer(
        model_type, model_name, tokenizer_name, num_classes, state_dict
):
    model_class = getattr(transformers, model_name)
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=None,
        config=model_type,
        num_labels=num_classes,
        state_dict=state_dict,
    )
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(model_type)

    return model, tokenizer


def load_binary_bert():
    state_dict = torch.hub.load_state_dict_from_url(
        "https://github.com/unitaryai/detoxify/releases/download/v0.1-alpha/toxic_bias-4e693588.ckpt")

    class_names = state_dict["config"]["dataset"]["args"]["classes"]

    model, tokenizer = get_model_and_tokenizer(
        **state_dict["config"]["arch"]["args"], state_dict=state_dict["state_dict"]
    )

    return model, tokenizer, class_names
