from __future__ import print_function
from keras import backend as K
from keras.layers import Input, Convolution1D, AveragePooling1D, GlobalAveragePooling1D, Dense, Lambda, merge, Embedding, TimeDistributed, RepeatVector, Permute
from keras.models import Model
# from keras.utils.visualize_util import plot
import numpy as np


def compute_match_score(l_r):
    return 1. / (
        1. +
        K.sqrt(
            -2 * K.batch_dot(l_r[0], l_r[1], [2, 2]) +
            K.expand_dims(K.sum(K.square(l_r[0]), axis=2), -1) +
            K.expand_dims(K.sum(K.square(l_r[1]), axis=2), 1)
        )
    )


def MatchScore(l, r):
    # dot = merge([l, r], mode="dot")
    return merge(
        [l, r],
        mode=compute_match_score,
        output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1])
    )


def just_match_score(left_seq_len, right_seq_len, embed_dimensions):
    lin = Input(shape=(left_seq_len, embed_dimensions))
    rin = Input(shape=(right_seq_len, embed_dimensions))
    matchscore = MatchScore(lin, rin)
    return Model([lin, rin], matchscore)


def ABCNN(
        left_seq_len, right_seq_len, vocab_size, embed_dimensions, nb_filter, filter_width,
        pool_length=2, conv_depth=1, dropout=0.4, abcnn_1=True, abcnn_2=True, collect_sentence_representations=False
):
    left_input = Input(shape=(left_seq_len,))
    right_input = Input(shape=(right_seq_len,))

    left_embed = Embedding(input_dim=vocab_size, output_dim=embed_dimensions, dropout=dropout)(left_input)
    right_embed = Embedding(input_dim=vocab_size, output_dim=embed_dimensions, dropout=dropout)(right_input)

    if abcnn_1:
        match_score = MatchScore(left_embed, right_embed)

        # compute attention
        attention_left = TimeDistributed(Dense(embed_dimensions), input_shape=(left_seq_len, right_seq_len))(match_score)
        match_score_t = Permute((2, 1))(match_score)
        attention_right = TimeDistributed(Dense(embed_dimensions), input_shape=(right_seq_len, left_seq_len))(match_score_t)

        # apply attention TODO this should be stacked to form an "order 3 tensor" - how?
        left_embed = merge([left_embed, attention_left], mode="mul")
        right_embed = merge([right_embed, attention_right], mode="mul")

    left_sentence_representations = []
    right_sentence_representations = []

    pool_left = left_embed
    pool_right = right_embed
    for i in range(conv_depth):
        # TODO should this be wide convolution?
        conv_left = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="same")(pool_left)
        conv_right = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="same")(pool_right)
        if abcnn_2:
            conv_match_score = MatchScore(conv_left, conv_right)

            # compute attention
            conv_attention_left = Lambda(
                lambda match: K.sum(match, axis=-1),
                output_shape=(conv_match_score._keras_shape[1],)
            )(conv_match_score)
            conv_attention_right = Lambda(
                lambda match: K.sum(match, axis=-2),
                output_shape=(conv_match_score._keras_shape[2],)
            )(conv_match_score)
            left_conv_attn_mask_t = RepeatVector(nb_filter)(conv_attention_left)
            right_conv_attn_mask_t = RepeatVector(nb_filter)(conv_attention_right)

            conv_attention_left = Permute((2, 1))(left_conv_attn_mask_t)
            conv_attention_right = Permute((2, 1))(right_conv_attn_mask_t)

            # apply attention
            conv_left = merge([conv_left, conv_attention_left], mode="mul")
            conv_right = merge([conv_right, conv_attention_right], mode="mul")

        pool_left = AveragePooling1D(pool_length=pool_length)(conv_left)
        pool_right = AveragePooling1D(pool_length=pool_length)(conv_right)

        if collect_sentence_representations or i == conv_depth - 1:  # always collect last layers global representation
            left_sentence_representations.append(GlobalAveragePooling1D()(conv_left))
            right_sentence_representations.append(GlobalAveragePooling1D()(conv_right))

    left_sentence_rep = left_sentence_representations.pop(-1)
    if left_sentence_representations:
        left_sentence_rep = merge([left_sentence_rep] + left_sentence_representations, mode="concat")

    right_sentence_rep = right_sentence_representations.pop(-1)
    if right_sentence_representations:
        right_sentence_rep = merge([right_sentence_rep] + right_sentence_representations, mode="concat")

    global_representation = merge([left_sentence_rep, right_sentence_rep], mode="concat")

    classify = Dense(1, activation="sigmoid")(global_representation)

    return Model([left_input, right_input], output=classify)


def test_matchscore():
    left = np.random.random((num_samples, left_seq_len, embed_dimensions))
    right = np.random.random((num_samples, right_seq_len, embed_dimensions))
    model = just_match_score(left_seq_len, right_seq_len, embed_dimensions)

    model.compile(optimizer="sgd", loss="categorical_crossentropy")
    # model.fit(X, y)
    res = model.predict([left, right])
    print(res)
    print(res.shape)
    test = 1. / (1. + np.sqrt(
        -2 * np.dot(left[0], right[0].T) +
        np.expand_dims(np.sum(np.square(left[0]), axis=1), -1) +
        np.expand_dims(np.sum(np.square(right[0]), axis=1), 0)
    ))
    print(res[0] - test)
    print(test.shape)
    print(np.sum(test, axis=-1).shape)
    print(np.sum(test, axis=-2).shape)


if __name__ == "__main__":
    num_samples = 1000
    vocab_size = 3500

    left_seq_len = 12
    right_seq_len = 8

    embed_dimensions = 50

    nb_filter = 64
    filter_width = 3

    X = [
        np.random.randint(0, vocab_size, (num_samples, left_seq_len,)),
        np.random.randint(0, vocab_size, (num_samples, right_seq_len,))
    ]
    Y = np.random.randint(0, 1, (num_samples,))

    model = ABCNN(
        left_seq_len=left_seq_len, right_seq_len=right_seq_len, vocab_size=vocab_size, conv_depth=2,
        embed_dimensions=embed_dimensions, nb_filter=nb_filter, filter_width=filter_width,
        collect_sentence_representations=True
    )

    model.compile(optimizer="sgd", loss="mse")
    # plot(model, to_file="abcnn.svg")
    model.fit(X, Y)




