import transformers
import msgpack

def get_model_vocab(model_name="StanfordShahLab/clmbr-t-base",
                    codes_only=False):
    
    """
    Get the vocabulary of a model.

    Args:
        model_name: The name of the model to get the vocabulary from.
        codes_only: Whether to return only the codes or the entire vocabulary.

    Returns:
        The vocabulary of the model.

    Example:
        >>> model_codes  = get_model_vocab(codes_only=True)
        >>> print(len(model_codes))
    """
    
    dictionary_file = transformers.utils.hub.cached_file(
                model_name, "dictionary.msgpack",
            )

    with open(dictionary_file, "rb") as f:
        dictionary = msgpack.load(f)

    vocab = dictionary["vocab"]
    if codes_only:
        codes = []
        for i, dict_entry in enumerate(vocab):
            if dict_entry["type"] == "code":
                codes.append(dict_entry["code_string"])
        return codes
    else:
        return vocab
    
