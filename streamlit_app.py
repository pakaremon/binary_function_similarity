import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM

# Function to create Siamese Model
def create_siamese_model(input_shape):
    shared_lstm = LSTM(64, return_sequences=False)
    
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)
    
    output_1 = shared_lstm(input_1)
    output_2 = shared_lstm(input_2)
    
    combined = tf.keras.layers.concatenate([output_1, output_2])
    combined = Dense(64, activation='relu')(combined)
    combined = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[input_1, input_2], outputs=combined)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Function to train the model
def train_model(X1_train, X2_train, y_train):
    input_shape = X1_train.shape[1:]
    model = create_siamese_model(input_shape)
    
    history = model.fit([X1_train, X2_train], y_train, epochs=10, batch_size=32, validation_split=0.2)
    model.save('siamese_model.h5')
    
    return history

# Streamlit app
st.title('Siamese Network Training App')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    import pandas as pd
    data = pd.read_csv(uploaded_file)
    
    # Assuming the CSV has columns X1, X2, and y
    X1 = np.array(data['X1'].tolist())
    X2 = np.array(data['X2'].tolist())
    y = np.array(data['y'].tolist())

    X1_train = np.array([list(map(float, x.split())) for x in X1])
    X2_train = np.array([list(map(float, x.split())) for x in X2])
    y_train = y.astype(int)
    
    st.write('Data Loaded Successfully')

    if st.button('Train Model'):
        history = train_model(X1_train, X2_train, y_train)
        st.success('Model trained successfully')

        # Display the training history
        st.write('Training History')
        st.line_chart(history.history['accuracy'])
        st.line_chart(history.history['loss'])
