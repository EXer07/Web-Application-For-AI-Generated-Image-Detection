import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
# ฟังก์ชันสำหรับการโหลดโมเดลจากไฟล์ .h5
def load_model_from_h5():
    model = tf.keras.models.load_model('/Users/titiphonphunmongkon/Documents/Web-Application-For-AI-Generated-Image-Detection/ResNet50_finetune_model.h5')  # ระบุพาธไฟล์ .h5 ที่บันทึกโมเดลไว้
    return model

# ฟังก์ชันสำหรับการพรีโปรเซสภาพ
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image ให้ตรงกับ input_shape ของโมเดล
    img = np.array(img)  # แปลงจาก PIL image เป็น numpy array
    img = img / 255.0  # Normalize ค่าพิกเซล
    img = np.expand_dims(img, axis=0)  # เพิ่ม batch dimension
    return img

# ฟังก์ชันหลักสำหรับแสดงแอป Streamlit
def main():
    st.title("AI Generated vs Real Image Classifier")
    
    # โหลดโมเดลจากไฟล์ .h5
    model = load_model_from_h5()
    
    # ให้ผู้ใช้เลือกไฟล์ภาพ
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # เปิดภาพที่อัพโหลด
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image.", use_column_width=True)
        
        # พรีโปรเซสภาพ
        img = preprocess_image(img)
        
        # ทำนายผล
        prediction = model.predict(img)
        prediction = "AI Generated" if prediction < 0.5 else "Real"
        
        # แสดงผลลัพธ์การทำนาย
        st.write(f"Prediction: {prediction}")

# รันแอป
if __name__ == "__main__":
    main()
