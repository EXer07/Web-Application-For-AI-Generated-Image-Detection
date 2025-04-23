import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
# load model
def load_model_from_h5():
    model = tf.keras.models.load_model('ResNet50_finetune_model.h5')  
    return model

# preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image
    img = np.array(img)  
    img = img / 255.0  # Normalize 
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img

# streamlit
def main():
    st.title("AI Generated vs Real Image Classifier")
    
   
    model = load_model_from_h5()
    
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image.", use_column_width=True)
        
       
        img = preprocess_image(img)
        
       
        prediction = model.predict(img)
        prediction = "AI Generated" if prediction < 0.5 else "Real"
        
        
        st.write(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
