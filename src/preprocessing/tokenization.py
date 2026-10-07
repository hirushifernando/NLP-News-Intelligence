def text_to_sequence(text, vocab):
    words = text.split()

    sequence = []

    for word in words:
        if word in vocab:
            sequence.append(vocab[word])
        else:
            sequence.append(vocab["<unk>"])

    return sequence


def pad_sequence(sequence, max_length):
    if len(sequence) > max_length:
        return sequence[:max_length]

    if len(sequence) < max_length:
        sequence = sequence + [0] * (max_length - len(sequence))

    return sequence