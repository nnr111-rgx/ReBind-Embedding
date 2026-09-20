from rapidfuzz.distance import Levenshtein


def bits_to_string(x):
    return "".join(str(int(v)) for v in x)


def edit_distance(a, b):
    return int(
        Levenshtein.distance(
            bits_to_string(a),
            bits_to_string(b),
        )
    )
