import torch.nn as nn


def random_initialization(model, tokenizer, backbone):
    ids = []
    for x in range(30000):
        tokenized_ids = tokenizer.encode(str(x))
        if 3 in tokenized_ids:
            tokenized_ids.remove(3)
        if 1 in tokenized_ids:
            tokenized_ids.remove(1)
        ids += tokenized_ids
    ids = list(set(ids))
    for index in ids:
        if "t5" in backbone:
            model.shared.weight.data[index] = nn.init.normal_(
                model.shared.weight.data[index], 0, 1.0
            )
        elif "llama" in backbone.lower():
            model.model.embed_tokens.weight.data[index] = nn.init.normal_(
                model.model.embed_tokens.weight.data[index], 0, 1.0
            )

    return model
