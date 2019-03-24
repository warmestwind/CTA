import tensorflow as tf
from keras.models import *
from keras.layers import Input, Conv3D, UpSampling3D, BatchNormalization, Activation, add, concatenate

# Identity Mappings in Deep Residual Networks
# Road Extraction by Deep Residual U-Net

scale = 4
active = 'relu'
class BatchNorm(BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training= True):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def res_block(x, nb_filters, strides):
    #             [128, 128], [(2, 2, 2), (1, 1, 1)
    res_path = BatchNorm()(x)
    res_path = Activation(activation=active)(res_path)
    # down sampling /2
    res_path = Conv3D(filters=nb_filters[0], kernel_size=(3,3,3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNorm()(res_path)
    res_path = Activation(activation=active)(res_path)
    res_path = Conv3D(filters=nb_filters[1], kernel_size=(3,3,3), padding='same', strides=strides[1])(res_path)
    shortcut = Conv3D(nb_filters[1], kernel_size=(1,1,1), strides=strides[0])(x)
    shortcut = BatchNorm()(shortcut)

    res_path = add([shortcut, res_path]) # c = nb[1]
    return res_path


def encoder(x):
    to_decoder = []

    main_path = Conv3D(filters=64//scale, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(x)
    main_path = BatchNorm()(main_path)

    main_path = Activation(activation=active)(main_path)

    main_path = Conv3D(filters=64//scale, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(main_path)

    shortcut = Conv3D(filters=64//scale, kernel_size=(1,1,1), strides=(1,1,1))(x)
    shortcut = BatchNorm()(shortcut)

    main_path = add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [128//scale, 128//scale], [(2,2,2), (1,1,1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256//scale, 256//scale], [(2,2,2), (1,1,1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder):
    main_path = UpSampling3D(size=(2,2,2))(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=-1)
    main_path = res_block(main_path, [256//scale, 256//scale], [(1,1,1), (1,1,1)])

    main_path = UpSampling3D(size=(2,2,2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=-1)
    main_path = res_block(main_path, [128//scale, 128//scale], [(1,1,1), (1,1,1)])

    main_path = UpSampling3D(size=(2,2,2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=-1)
    main_path = res_block(main_path, [64//scale, 64//scale], [(1,1,1), (1,1,1)])

    return main_path


def build_res_unet(input_shape):
    inputs = Input(shape=input_shape)

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[2], [512//scale//2, 512//scale//2], [(2,2,2), (1,1,1)])

    path = decoder(path, from_encoder=to_decoder)

    'bg, true, false'
    path = Conv3D(filters=3, kernel_size=(1,1,1), activation='softmax')(path)

    return Model(input=inputs, output=path)

def Model_1_0(inputs):

    to_decoder = encoder(inputs)

    path = res_block(to_decoder[1], [512, 512], [(2,2,2), (1,1,1)])

    path = decoder(path, from_encoder=to_decoder)

    'bg, true, false'
    logits = Conv3D(filters=3, kernel_size=(1,1,1), activation='softmax')(path)

    return logits #shape [b,256,128,128,3]


#model = build_res_unet((256,128,128,1))
#Layer (type), Output Shape =, Param #, Connected to
#model.summary(positions=[.33, .6, .7, 1.])