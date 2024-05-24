def apply_cmvn(feats):
    feats = feats - feats.mean(1, keepdims=True)
    feats = feats / feats.std(1, keepdims=True)
    return feats