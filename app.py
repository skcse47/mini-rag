import math

A = [1, 2]
B = [2, -1]


dot = sum(a * b for a, b in zip(A, B))

mag_a = math.sqrt(sum(a * a for a in A))
mag_b = math.sqrt(sum(b * b for b in B))

cosine = dot / (mag_a * mag_b)

print(cosine)