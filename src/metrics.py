def precision_at_k(predicted, actual, k):
    predicted_k = set(predicted[:k])
    actual_set = set(actual)
    if k == 0:
        return 0.0
    return len(predicted_k & actual_set) / k

def recall_at_k(predicted, actual, k):
    predicted_k = set(predicted[:k])
    actual_set = set(actual)
    if len(actual) == 0:
        return 0.0
    return len(predicted_k & actual_set) / len(actual)

def f2_at_k(predicted, actual, k):
    p = precision_at_k(predicted, actual, k)
    r = recall_at_k(predicted, actual, k)
    if p + r == 0:
        return 0.0
    return 5 * (p * r) / (4 * p + r)