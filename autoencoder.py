
"""
Key Qs:
1) Which AE? Undercomplete (standard), sparse, or contractive
    - my rec: sparse or contractive
2) Doing convolutional AE?
    - my rec: yes
3) Structure: Where does AE need to "plug in" with DQN?
4) Visualization Qs:
        a) Visualize using same tools as BD's CNN?
            - my rec: yes, if possible
        b) If not, what do we envision visualization output looking like? What is "shape"?
            - my rec: same-size image as Pong board
        c) Which layer do we want to visualize?
            - my rec: middle layer
"""

# LOOK AT REGULARIZED (ESP. SPARSE) AUTOENCODERS FOR VISUALIZATION?
# LOOK AT CONTRACTIVE AND/OR SPARSE AUTOENCODERS?

# SEE ABOUT GENERAL CNN OR NN VISUALIZATION
# CHECK OUT CONTRACTIVE AUTOENCODERS

## MULTILAYER AE
input_size = 784
hidden_size = 128
code_size = 64

x = Input(shape=(input_size,))

# Encoder
hidden_1 = Dense(hidden_size, activation='relu')(x)
h = Dense(code_size, activation='relu')(hidden_1)

# Decoder
hidden_2 = Dense(hidden_size, activation='relu')(h)
r = Dense(input_size, activation='sigmoid')(hidden_2)

autoencoder = Model(input=x, output=r)
autoencoder.compile(optimizer='adam', loss='mse')


# CONVOLUTIONAL AE

x = Input(shape=(28, 28,1))

# Encoder
conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2)
conv1_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
h = MaxPooling2D((2, 2), padding='same')(conv1_3)


# Decoder
conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
up1 = UpSampling2D((2, 2))(conv2_1)
conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D((2, 2))(conv2_2)
conv2_3 = Conv2D(16, (3, 3), activation='relu')(up2)
up3 = UpSampling2D((2, 2))(conv2_3)
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up3)

autoencoder = Model(input=x, output=r)
autoencoder.compile(optimizer='adam', loss='mse')


# REGULARIZED AE
input_size = 784
hidden_size = 64
output_size = 784

x = Input(shape=(input_size,))

# Encoder
h = Dense(hidden_size, activation='relu', activity_regularizer=regularizers.l1(10e-5))(x)

# Decoder
r = Dense(output_size, activation='sigmoid')(h)

autoencoder = Model(input=x, output=r)
autoencoder.compile(optimizer='adam', loss='mse')


#PLOT WEIGHTS OF AUTOENCODER
# https://www.mathworks.com/help/deeplearning/ref/autoencoder.plotweights.html#buy18fl-1
hiddenSize = 25;
autoenc = trainAutoencoder(X,hiddenSize, ...
  'L2WeightRegularization',0.004, ...
  'SparsityRegularization',4, ...
  'SparsityProportion',0.2);

plotWeights(autoenc)