import numpy as np

num_samples = 200
vocab_size = 3500
left_seq_len = 12
right_seq_len = 8
embed_dimensions = 50
nb_filter = 64
filter_width = 3
np.random.random()

left = np.random.random((num_samples, left_seq_len, 100))
right = np.random.random((num_samples, right_seq_len, 100))

l = left[0]
print(l.shape)
r = right[0]
print(r.shape)

r1 = np.sqrt(-2 * np.dot(l, r.T) + (l * l).sum(axis=1)[:, None] + (r * r).sum(axis=1)[None, :])
r2 = np.sqrt(
    -2 * np.dot(l, r.T) +
    np.expand_dims(np.sum(np.square(l), axis=1), -1) +
    np.expand_dims(np.sum(np.square(r), axis=1), 0)
)

print(r1 == r2)

print(r1.shape)
dots = np.zeros((num_samples, left_seq_len, right_seq_len))
dists = np.zeros((num_samples, left_seq_len, right_seq_len))
for i, (l, r) in enumerate(zip(left, right)):
    dots[i] = np.dot(l, r.T)
    dists[i] = np.sqrt(
        -2 * np.dot(l, r.T) +
        np.expand_dims(np.sum(np.square(l), axis=1), -1) +
        np.expand_dims(np.sum(np.square(r), axis=1), 0)
    )
print(dists[0] == r2)
print("Computed with loop")


def euclidean(l, r):
    return np.sqrt(
        -2 * np.dot(l, r.T) +
        np.expand_dims(np.sum(np.square(l), axis=1), 1) +
        np.expand_dims(np.sum(np.square(r), axis=1), 0)
    )


def cosine(l, r):
    return (
        1. + (
            np.dot(l, r) /
            (K.sqrt(K.sum(K.square(l), axis=2)) * K.sqrt(K.sum(K.square(r), axis=2)))
        ) / 2.
    )

res = np.tensordot(left, right, axes=[2, 2])
# res = euclidean(left, right)

print("Want shape: ", (dots.shape))
print("Got shape: ", res.shape)

