from __future__ import print_function
from keras import backend as K
from keras.layers import Input, merge
from keras.models import Model
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity


def compute_euclidean_match_score(l_r):
    l, r = l_r
    denominator = 1. + K.sqrt(
        -2 * K.batch_dot(l, r, axes=[2, 2]) +
        K.expand_dims(K.sum(K.square(l), axis=2), 2) +
        K.expand_dims(K.sum(K.square(r), axis=2), 1)
    )
    denominator = K.maximum(denominator, K.epsilon())
    return 1. / denominator


def compute_cos_match_score(l_r):
    # K.batch_dot(
    #     K.l2_normalize(l, axis=-1),
    #     K.l2_normalize(r, axis=-1),
    #     axes=[2, 2]
    # )

    l, r = l_r
    denominator = K.sqrt(K.batch_dot(l, l, axes=[2, 2]) *
                         K.batch_dot(r, r, axes=[2, 2]))
    denominator = K.maximum(denominator, K.epsilon())
    output = K.batch_dot(l, r, axes=[2, 2]) / denominator
    # output = K.expand_dims(output, 1)
    # denominator = K.maximum(denominator, K.epsilon())
    return output


def MatchScore(l, r, use_fn=compute_cos_match_score):
    return merge(
        [l, r],
        mode=use_fn,
        output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1])
    )


def euclidean_match_fn(left_seq_len, right_seq_len, embed_dimensions):
    lin = Input(shape=(left_seq_len, embed_dimensions))
    rin = Input(shape=(right_seq_len, embed_dimensions))
    matchscore = MatchScore(lin, rin, use_fn=compute_euclidean_match_score)
    return Model([lin, rin], matchscore)


def cos_match_fn(left_seq_len, right_seq_len, embed_dimensions):
    lin = Input(shape=(left_seq_len, embed_dimensions))
    rin = Input(shape=(right_seq_len, embed_dimensions))
    matchscore = MatchScore(lin, rin, use_fn=compute_cos_match_score)
    return Model([lin, rin], matchscore)


def test_matchscore():
    num_samples = 210

    left_seq_len = 12
    right_seq_len = 8

    embed_dimensions = 50

    left = np.random.random((num_samples, left_seq_len, embed_dimensions))
    right = np.random.random((num_samples, right_seq_len, embed_dimensions))

    model = euclidean_match_fn(left_seq_len, right_seq_len, embed_dimensions)
    model.compile(optimizer="sgd", loss="categorical_crossentropy")

    model2 = cos_match_fn(left_seq_len, right_seq_len, embed_dimensions)
    model2.compile(optimizer="sgd", loss="categorical_crossentropy")
    res = model.predict([left, right])
    res2 = model2.predict([left, right])

    print("############### euclid: res ~= test:")
    test_euclid = 1. / (1. + euclidean_distances(left[0], right[0]))

    print(np.isclose(res[0], test_euclid))
    print(test_euclid.shape)

    print("############### cos: res ~= test:")
    test_cos = cosine_similarity(left[0], right[0])

    print(np.isclose(res2[0], test_cos))
    print(test_cos.shape)


if __name__ == "__main__":
    test_matchscore()
