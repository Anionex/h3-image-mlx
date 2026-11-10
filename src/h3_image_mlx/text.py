from .core import Weights, attention, heads, mx, norm, rope
from tokenizers import Tokenizer


def encode(path, tokenizer_path, prompt, refs=()):
    tok = Tokenizer.from_file(str(tokenizer_path))
    ids = tok.encode(prompt, add_special_tokens=False).ids[:4096]
    if not ids:
        raise ValueError("Empty prompt")
    w = Weights(path)
    if refs:
        from .vision import prepare_prompt

        x, angles, rows, deep, tags = prepare_prompt(w, tok, prompt, refs)
    else:
        x = mx.stack([w._read("model.embed_tokens.weight", row=i) for i in ids])
        tags = mx.ones((len(ids),), dtype=mx.int32)
    for i in range(50):
        prefix = f"model.layers.{i}."
        h = norm(x, w[prefix + "input_layernorm.weight"], 1e-6)
        att = prefix + "self_attn."

        q, k, v = (
            heads(w.linear(att + name + "_proj", h), 128) for name in ("q", "k", "v")
        )
        q, k = (
            norm(value, w[att + name + "_norm.weight"], 1e-6)
            for value, name in ((q, "q"), (k, "k"))
        )
        if refs:
            q, k = rope(q, angles), rope(k, angles)
        else:
            q, k = (
                mx.fast.rope(
                    v, 128, traditional=False, base=5_000_000, scale=1.0, offset=0
                )
                for v in (q, k)
            )
        x = x + w.linear(att + "o_proj", attention(q, k, v, mask="causal"))
        h = norm(x, w[prefix + "post_attention_layernorm.weight"], 1e-6)
        mlp = prefix + "mlp."
        g = w.linear(mlp + "gate_proj", h)
        x = x + w.linear(
            mlp + "down_proj", (g * mx.sigmoid(g)) * w.linear(mlp + "up_proj", h)
        )
        if refs and i < len(deep):
            x = x.at[rows].add(deep[i].astype(x.dtype))
        mx.eval(x)
    return x, tags, w.int8_calls
