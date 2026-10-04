def count_common(words):
    counts = {}
    first_index = {}
    for idx, word in enumerate(words):
        if word not in counts:
            counts[word] = 1
            first_index[word] = idx
        else:
            counts[word] += 1
    items = [(word, counts[word], first_index[word]) for word in counts]
    items.sort(key=lambda x: (-x[1], x[2]))
    top = items[:4]
    return [(word, count) for word, count, _ in top]
