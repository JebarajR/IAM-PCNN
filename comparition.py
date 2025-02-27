n_features = 150
from tensorflow import keras
from other.LinkNet import *
from other.dense_net import *
from other.Squeezenet import *
from other.EfficientNet import *
from other.Deep_Residual_network import *
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

ep = 25
n = [0, 1, 2, 3]
from other.Confusion_matrix import multi_confu_matrix
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, SimpleRNN, Bidirectional, LSTM


def comp(X_train, X_test, Y_train, Y_test):
    def CNN_classify():
        train_data = X_train.astype('float32')
        test_data = X_test.astype('float32')
        train_data = train_data / np.max(train_data)
        test_data = test_data / np.max(test_data)
        train_x = train_data[:, :, np.newaxis]
        train_y = keras.utils.to_categorical(Y_train)
        test_x = test_data[:, :, np.newaxis]
        test_y = keras.utils.to_categorical(Y_test)
        # evaluate model
        verbose, epochs, batch_size, n = 0, 1, 32, 2
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        model = Sequential()
        model.add(Conv1D(32, kernel_size=1, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(Conv1D(64, kernel_size=1, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())
        ln = len(np.unique(Y_train))
        model.add(Dense(ln, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        # fit network
        history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0)
        y_predict = np.argmax(model.predict(test_x), axis=-1)
        y_pred = array(y_predict, axis=[2, 1])
        return y_pred

    def con_pcnn():
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras import layers, models

        import tensorflow.keras.backend as K
        class attention(layers.Layer):
            def __init__(self, **kwargs):
                super(attention, self).__init__(**kwargs)

            def build(self, input_shape):
                self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                         initializer='random_normal', trainable=True)
                self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                         initializer='zeros', trainable=True)
                super(attention, self).build(input_shape)

            def call(self, x):
                # Alignment scores. Pass them through tanh function
                e = K.tanh(K.dot(x, self.W) + self.b)
                # Remove dimension of size 1
                e = K.squeeze(e, axis=-1)
                # Compute the weights
                alpha = K.softmax(e)
                print('alpha:', alpha)
                print('x', x)
                # Reshape to tensorFlow format
                alpha = K.expand_dims(alpha, axis=-1)
                # Compute the context vector
                context = x * alpha
                context = K.sum(context, axis=1)
                return context

        def proposed_parallel_cnn(X_train, X_test, Y_train, Y_test, db):
            # Custom Z-score Standardization Layer
            class ZScoreNormalization(layers.Layer):
                def call(self, inputs):
                    mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
                    stddev = tf.math.reduce_std(inputs, axis=0, keepdims=True)
                    alpha = 1 / inputs - mean
                    return (inputs - mean) / (stddev + tf.keras.backend.epsilon()) * alpha

            # Custom ReLU Activation Function
            def proposed_relu(x):
                return tf.where(x > 0, x, x * tf.nn.sigmoid(x))

            # Custom Batch Normalization Layer
            class CustomBatchNormalization(layers.BatchNormalization):
                def call(self, inputs, training=False):
                    return super().call(inputs, training=training)

            def build_pcnn(input_shape_1d, input_shape_2d):
                # 1D-CNN branch
                input_1d = layers.Input(shape=input_shape_1d)
                x1 = layers.Conv1D(32, kernel_size=3)(input_1d)
                x1 = layers.Activation(proposed_relu)(x1)
                x1 = layers.MaxPooling1D(pool_size=2)(x1)

                x1 = layers.Conv1D(64, kernel_size=3)(x1)
                x1 = layers.Activation(proposed_relu)(x1)
                x1 = layers.MaxPooling1D(pool_size=2)(x1)

                # Add attention layer for 1D branch
                x1 = attention()(x1)

                x1 = layers.Flatten()(x1)

                # 2D-CNN branch
                input_2d = layers.Input(shape=input_shape_2d)
                x2 = layers.Conv2D(32, kernel_size=(3, 3))(input_2d)
                x2 = layers.Activation(proposed_relu)(x2)
                x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)

                x2 = layers.Conv2D(64, kernel_size=(3, 3))(x2)
                x2 = layers.Activation(proposed_relu)(x2)
                x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)

                # Add attention layer for 2D branch
                x2 = attention()(x2)

                x2 = layers.Flatten()(x2)

                # Concatenate the outputs from both branches
                combined = layers.concatenate([x1, x2])

                # Fully connected layers
                x = layers.Dense(128)(combined)
                x = layers.Activation(proposed_relu)(x)
                x = layers.Dropout(0.5)(x)
                output = layers.Dense(10, activation='softmax')(x)  # Change 10 to the number of classes

                # Create the model
                model = models.Model(inputs=[input_1d, input_2d], outputs=output)

                return model

            # Example input shapes
            input_shape_1d = (100, 1)  # Example shape for 1D-CNN (time series)
            input_shape_2d = (64, 64, 1)  # Example shape for 2D-CNN (time-frequency data)

            # Build the model
            model = build_pcnn(input_shape_1d, input_shape_2d)

            # Compile the model
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Summary of the model
            model.summary()

            # Prepare data for training
            x_train_1d = np.resize(X_train, (X_train.shape[0], 100, 1))
            x_train_2d = np.resize(X_train, (X_train.shape[0], 64, 64, 1))
            y_train = Y_train
            x_test_1d = np.resize(X_test, (X_test.shape[0], 100, 1))
            x_test_2d = np.resize(X_test, (X_test.shape[0], 64, 64, 1))

            # Fit the model
            model.fit([x_train_1d, x_train_2d], y_train, epochs=1, batch_size=32, verbose=False)
            y_pred = np.argmax(model.predict([x_test_1d, x_test_2d]), axis=-1)
            pred = array(y_pred, [1, 1])
            return pred

    def rnn():
        train_X = X_train.reshape(-1, X_train.shape[1], 1)
        test_X = X_test.reshape(-1, X_test.shape[1], 1)
        model = Sequential()
        model.add(SimpleRNN(64, input_shape=train_X[0].shape, activation='relu'))
        ln = len(np.unique(Y_train))
        model.add(Dense(ln, activation='sigmoid'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(train_X, Y_train, epochs=1, batch_size=10, verbose=0)
        y_predict = np.argmax(model.predict(test_X), axis=-1)
        y_pred = array(y_predict, axis=[2, 1])
        return y_pred

    def Efficient_Net():
        return Efficient_net_process(X_train, Y_train, X_test, Y_test)

    def LinkNet():
        return linknet(X_train, X_test, Y_train, Y_test)

    def den():
        return dense_net_classify(X_train, X_test, Y_train, Y_test)

    def sqez():
        return csqueezenet_(X_train, Y_train, X_test, Y_test)

    def resnet():
        pred = classify(X_train, Y_train, X_test, Y_test)
        return pred

    return [Efficient_Net(), LinkNet(), den(), rnn(), sqez(), resnet(), CNN_classify()]


def proposed_method(X_train, X_test, Y_train, Y_test):
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models

    import tensorflow.keras.backend as K
    class attention(layers.Layer):
        def __init__(self, **kwargs):
            super(attention, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                     initializer='random_normal', trainable=True)
            self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                     initializer='zeros', trainable=True)
            super(attention, self).build(input_shape)

        def call(self, x):
            # Alignment scores. Pass them through tanh function
            e = K.tanh(K.dot(x, self.W) + self.b)
            # Remove dimension of size 1
            e = K.squeeze(e, axis=-1)
            # Compute the weights
            alpha = K.softmax(e)
            print('alpha:', alpha)
            print('x', x)
            # Reshape to tensorFlow format
            alpha = K.expand_dims(alpha, axis=-1)
            # Compute the context vector
            context = x * alpha
            context = K.sum(context, axis=1)
            return context

    # Custom ReLU Activation Function
    def proposed_relu(x):
        return tf.where(x > 0, x, x * tf.nn.sigmoid(x))

    def build_pcnn(input_shape_1d, input_shape_2d):
        # 1D-CNN branch
        input_1d = layers.Input(shape=input_shape_1d)
        x1 = layers.Conv1D(32, kernel_size=3)(input_1d)
        x1 = layers.Activation(proposed_relu)(x1)
        x1 = layers.MaxPooling1D(pool_size=2)(x1)

        x1 = layers.Conv1D(64, kernel_size=3)(x1)
        x1 = layers.Activation(proposed_relu)(x1)
        x1 = layers.MaxPooling1D(pool_size=2)(x1)

        # Add attention layer for 1D branch
        x1 = attention()(x1)

        x1 = layers.Flatten()(x1)

        # 2D-CNN branch
        input_2d = layers.Input(shape=input_shape_2d)
        x2 = layers.Conv2D(32, kernel_size=(3, 3))(input_2d)
        x2 = layers.Activation(proposed_relu)(x2)
        x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)

        x2 = layers.Conv2D(64, kernel_size=(3, 3))(x2)
        x2 = layers.Activation(proposed_relu)(x2)
        x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)

        # Add attention layer for 2D branch
        x2 = attention()(x2)

        x2 = layers.Flatten()(x2)

        # Concatenate the outputs from both branches
        combined = layers.concatenate([x1, x2])

        # Fully connected layers
        x = layers.Dense(128)(combined)
        x = layers.Activation(proposed_relu)(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(10, activation='softmax')(x)  # Change 10 to the number of classes

        # Create the model
        model = models.Model(inputs=[input_1d, input_2d], outputs=output)

        return model

    # Example input shapes
    input_shape_1d = (100, 1)  # Example shape for 1D-CNN (time series)
    input_shape_2d = (64, 64, 1)  # Example shape for 2D-CNN (time-frequency data)

    # Build the model
    model = build_pcnn(input_shape_1d, input_shape_2d)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    # Prepare data for training
    x_train_1d = np.resize(X_train, (X_train.shape[0], 100, 1))
    x_train_2d = np.resize(X_train, (X_train.shape[0], 64, 64, 1))
    y_train = Y_train
    x_test_1d = np.resize(X_test, (X_test.shape[0], 100, 1))
    x_test_2d = np.resize(X_test, (X_test.shape[0], 64, 64, 1))

    # Fit the model
    model.fit([x_train_1d, x_train_2d], y_train, epochs=1, batch_size=32, verbose=False)
    y_pred = np.argmax(model.predict([x_test_1d, x_test_2d]), axis=-1)
    pred = array(y_pred, [1, 1])
    return pred


def AB_prop_(X_train, X_test, Y_train, Y_test, i):
    if i == 0:
        ll = 0
    elif i == 1:
        ll = 2
    else:
        ll = 3
    pred2 = proposed_method(X_train, X_test, Y_train, Y_test)
    pred = array(pred2, axis=[ll, 1])
    out = multi_confu_matrix(Y_test, pred)
    return out[0]


def ab_comp(feat, label, cond):
    c1 = []
    feat1 = feat
    feat1 = feat1
    f = [feat, feat1, feat, feat1]
    for i in range(0, 4):
        fet = f[i]
        lp = 0.9
        X_train, X_test, Y_train, Y_test = train_test_split(fet, label, test_size=lp, random_state=0)
        np.save('pre_evaluated/Y_test', Y_test)
        np.save('pre_evaluated/tp', lp)
        c1.append(AB_prop_(X_train, X_test, Y_train, Y_test, i))
    clmn = ['proposed with conv wiener', 'proposed with conv LGBPHS', 'proposed with conv pcnn',
            'proposed without feature extraction']
    indx = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr']
    import pandas as pd
    val = np.array(c1)
    d = pd.DataFrame(val.transpose(), columns=clmn, index=indx)
    d.to_csv(f'pre_evaluated/ablation-{cond}.csv')
    return d
