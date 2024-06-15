import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except OSError:
        st.error(f"Unable to load model from {model_path}. Please check the file path.")
        return None

def preprocess_image(image):
    image = image.resize((224, 224))  
    image = np.asarray(image) / 255.0 
    image = np.expand_dims(image, axis=0)  
    return image


def predict(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction


def main():
    st.title('Image Classifier')
    st.text('Upload an image for classification')

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        
        if st.button('Classify'):
            
            model_path = "C:/Users/amits/OneDrive/Documents/Model[1]/Model/EfficientNetmodel_weights2.h5" 
            model = load_model(model_path)

            if model is not None:
               
                prediction = predict(image, model)

               
                st.write(f'Prediction: {prediction}')


if __name__ == '__main__':
    main()
