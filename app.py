import math

A = [1, 2]
B = [2, 1]



def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))

    mag_a = math.sqrt(sum(a * a for a in vec1))
    mag_b = math.sqrt(sum(b * b for b in vec2))

    cosine = dot / (mag_a * mag_b)

    return cosine


print(cosine_similarity(A,B))