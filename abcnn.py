from __future__ import print_function
from keras import backend as K
from keras.layers import Input, Convolution1D, Convolution2D, AveragePooling1D, GlobalAveragePooling1D, Dense, Lambda, merge, Embedding, TimeDistributed, RepeatVector, Permute, ZeroPadding1D, ZeroPadding2D, Reshape
from keras.models import Model
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


def plot(*args, **kwargs):
    try:
        from keras.utils.visualize_util import plot as plt
        plt(*args, **kwargs)
    except:
        print("plot could not be imported, sorry.")
        pass


def compute_match_score(l_r_dot):  # TODO This is causing NaN values during backprop, find bug
    l, r, dot = l_r_dot
    return 1. / (
        1. +
        K.sqrt(
            -2 * dot +
            K.expand_dims(K.sum(K.square(l), axis=2), -1) +
            K.expand_dims(K.sum(K.square(r), axis=2), 1)
        )
    )


def MatchScore(l, r):
    dot = merge([l, r], mode="dot")
    return dot
    # return merge(  # TODO buggy
    #     [l, r, dot],
    #     mode=compute_match_score,
    #     output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1])
    # )


def just_match_score(left_seq_len, right_seq_len, embed_dimensions):
    lin = Input(shape=(left_seq_len, embed_dimensions))
    rin = Input(shape=(right_seq_len, embed_dimensions))
    matchscore = MatchScore(lin, rin)
    return Model([lin, rin], matchscore)


def ABCNN(
        left_seq_len, right_seq_len, vocab_size, embed_dimensions, nb_filter, filter_width,
        depth=2, dropout=0.4, abcnn_1=True, abcnn_2=True, collect_sentence_representations=False, aux_outputs=False,
        abcnn1_2D=True
):
    assert depth >= 1, "Need at least one layer to build ABCNN"
    assert not (depth == 1 and abcnn_2), "Cannot build ABCNN-2 with only one layer!"

    left_sentence_representations = []
    right_sentence_representations = []

    left_input = Input(shape=(left_seq_len,))
    right_input = Input(shape=(right_seq_len,))

    left_embed = Embedding(input_dim=vocab_size, output_dim=embed_dimensions, dropout=dropout)(left_input)
    right_embed = Embedding(input_dim=vocab_size, output_dim=embed_dimensions, dropout=dropout)(right_input)

    if abcnn_1:
        match_score = MatchScore(left_embed, right_embed)

        # compute attention
        attention_left = TimeDistributed(
            Dense(embed_dimensions, activation="relu"), input_shape=(left_seq_len, right_seq_len))(match_score)
        match_score_t = Permute((2, 1))(match_score)
        attention_right = TimeDistributed(
            Dense(embed_dimensions, activation="relu"), input_shape=(right_seq_len, left_seq_len))(match_score_t)

        left_reshape = Reshape((1, attention_left._keras_shape[1], attention_left._keras_shape[2]))
        right_reshape = Reshape((1, attention_right._keras_shape[1], attention_right._keras_shape[2]))

        attention_left = left_reshape(attention_left)
        left_embed = left_reshape(left_embed)

        attention_right = right_reshape(attention_right)
        right_embed = right_reshape(right_embed)

        # concat attention
        # (samples, channels, rows, cols)
        left_embed = merge([left_embed, attention_left], mode="concat", concat_axis=1)
        right_embed = merge([right_embed, attention_right], mode="concat", concat_axis=1)

        # Padding so we have wide convolution
        left_embed_padded = ZeroPadding2D((filter_width - 1, 0))(left_embed)
        right_embed_padded = ZeroPadding2D((filter_width - 1, 0))(right_embed)

        # 2D convolutions so we have the ability to treat channels. Effectively, we are still doing 1-D convolutions.
        conv_left = Convolution2D(
            nb_filter=nb_filter, nb_row=filter_width, nb_col=embed_dimensions, activation="tanh", border_mode="valid")(
            left_embed_padded
        )

        # Reshape and Permute to get back to 1-D
        conv_left = (Reshape((conv_left._keras_shape[1], conv_left._keras_shape[2])))(conv_left)
        conv_left = Permute((2, 1))(conv_left)

        conv_right = Convolution2D(
            nb_filter=nb_filter, nb_row=filter_width, nb_col=embed_dimensions, activation="tanh",
            border_mode="valid")(
            right_embed_padded
        )
        # Reshape and Permute to get back to 1-D
        conv_right = (Reshape((conv_right._keras_shape[1], conv_right._keras_shape[2])))(conv_right)
        conv_right = Permute((2, 1))(conv_right)

    else:
        # Padding so we have wide convolution
        left_embed_padded = ZeroPadding1D(filter_width - 1)(left_embed)
        right_embed_padded = ZeroPadding1D(filter_width - 1)(right_embed)
        conv_left = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(left_embed_padded)
        conv_right = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(right_embed_padded)

    pool_left = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_left)
    pool_right = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_right)

    assert pool_left._keras_shape[1] == left_seq_len
    assert pool_right._keras_shape[1] == right_seq_len

    if collect_sentence_representations or depth == 1:  # always collect last layers global representation
        left_sentence_representations.append(GlobalAveragePooling1D()(conv_left))
        right_sentence_representations.append(GlobalAveragePooling1D()(conv_right))

    # ###################### #
    # ### END OF ABCNN-1 ### #
    # ###################### #

    for i in range(depth - 1):
        pool_left = ZeroPadding1D(filter_width - 1)(pool_left)
        pool_right = ZeroPadding1D(filter_width - 1)(pool_right)
        # Wide convolution
        conv_left = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(pool_left)
        conv_right = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(pool_right)

        if abcnn_2:
            conv_match_score = MatchScore(conv_left, conv_right)

            # compute attention
            conv_attention_left = Lambda(lambda match: K.sum(match, axis=-1), output_shape=(conv_match_score._keras_shape[1],))(conv_match_score)
            conv_attention_right = Lambda(lambda match: K.sum(match, axis=-2), output_shape=(conv_match_score._keras_shape[2],))(conv_match_score)

            conv_attention_left = Permute((2, 1))(RepeatVector(nb_filter)(conv_attention_left))
            conv_attention_right = Permute((2, 1))(RepeatVector(nb_filter)(conv_attention_right))

            # apply attention  TODO is "multiply each value by the sum of it's respective attention row/column" correct?
            conv_left = merge([conv_left, conv_attention_left], mode="mul")
            conv_right = merge([conv_right, conv_attention_right], mode="mul")

        pool_left = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_left)
        pool_right = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_right)

        assert pool_left._keras_shape[1] == left_seq_len
        assert pool_right._keras_shape[1] == right_seq_len

        if collect_sentence_representations or (i == (depth - 2)):  # always collect last layers global representation
            left_sentence_representations.append(GlobalAveragePooling1D()(conv_left))
            right_sentence_representations.append(GlobalAveragePooling1D()(conv_right))

    # ###################### #
    # ### END OF ABCNN-2 ### #
    # ###################### #

    # Merge collected sentence representations if necessary
    left_sentence_rep = left_sentence_representations.pop(-1)
    if left_sentence_representations:
        left_sentence_rep = merge([left_sentence_rep] + left_sentence_representations, mode="concat")

    right_sentence_rep = right_sentence_representations.pop(-1)
    if right_sentence_representations:
        right_sentence_rep = merge([right_sentence_rep] + right_sentence_representations, mode="concat")

    global_representation = merge([left_sentence_rep, right_sentence_rep], mode="concat")

    # Add logistic regression on top.
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
    num_samples = 210
    vocab_size = 150

    left_seq_len = 12
    right_seq_len = 8

    embed_dimensions = 50

    nb_filter = 64
    filter_width = 3

    X = [
        np.random.randint(0, vocab_size, (num_samples, left_seq_len,)),
        np.random.randint(0, vocab_size, (num_samples, right_seq_len,))
    ]
    Y = np.random.randint(0, 2, (num_samples,))

    plot_all = False
    if plot_all:
        bcnn = ABCNN(
            left_seq_len=left_seq_len, right_seq_len=right_seq_len, vocab_size=vocab_size, depth=2,
            embed_dimensions=embed_dimensions, nb_filter=nb_filter, filter_width=filter_width,
            collect_sentence_representations=True, abcnn_1=False, abcnn_2=False
        )
        plot(bcnn, to_file="bcnn.svg")

        bcnn_deep_nocollect = ABCNN(
            left_seq_len=left_seq_len, right_seq_len=right_seq_len, vocab_size=vocab_size, depth=4,
            embed_dimensions=embed_dimensions, nb_filter=nb_filter, filter_width=filter_width,
            collect_sentence_representations=False, abcnn_1=False, abcnn_2=False
        )
        plot(bcnn_deep_nocollect, to_file="bcnn_deep_nocollect.svg")

        abcnn1 = ABCNN(
            left_seq_len=left_seq_len, right_seq_len=right_seq_len, vocab_size=vocab_size, depth=2,
            embed_dimensions=embed_dimensions, nb_filter=nb_filter, filter_width=filter_width,
            collect_sentence_representations=True, abcnn_1=True, abcnn_2=False
        )
        plot(abcnn1, to_file="abcnn1.svg")

        abcnn2 = ABCNN(
            left_seq_len=left_seq_len, right_seq_len=right_seq_len, vocab_size=vocab_size, depth=2,
            embed_dimensions=embed_dimensions, nb_filter=nb_filter, filter_width=filter_width,
            collect_sentence_representations=True, abcnn_1=False, abcnn_2=True
        )
        plot(abcnn2, to_file="abcnn2.svg")

        abcnn3 = ABCNN(
            left_seq_len=left_seq_len, right_seq_len=right_seq_len, vocab_size=vocab_size, depth=2,
            embed_dimensions=embed_dimensions, nb_filter=nb_filter, filter_width=filter_width,
            collect_sentence_representations=True, abcnn_1=True, abcnn_2=True
        )
        plot(abcnn3, to_file="abcnn3.svg")

        abcnn3_deep = ABCNN(
            left_seq_len=left_seq_len, right_seq_len=right_seq_len, vocab_size=vocab_size, depth=4,
            embed_dimensions=embed_dimensions, nb_filter=nb_filter, filter_width=filter_width,
            collect_sentence_representations=True, abcnn_1=True, abcnn_2=True
        )
        plot(abcnn3_deep, to_file="abcnn3-deep.svg")

    model = ABCNN(
        left_seq_len=left_seq_len, right_seq_len=right_seq_len, vocab_size=vocab_size, depth=2,
        embed_dimensions=embed_dimensions, nb_filter=nb_filter, filter_width=filter_width,
        collect_sentence_representations=True, abcnn_1=True, abcnn_2=True
    )

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    model.fit(X, Y, nb_epoch=3)
    print(model.predict(X)[0])
