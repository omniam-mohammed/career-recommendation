import streamlit as st
import numpy as np
import joblib
from PIL import Image
import os
import traceback

# تحميل النماذج باستخدام مسارات نسبية
try:
    scaler = joblib.load("./Models/scaler.pkl")
    pca = joblib.load("./Models/pca.pkl")
    selector = joblib.load("./Models/selector.pkl")
    model = joblib.load("./Models/model.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# قائمة أسماء الوظائف
class_names = [
    'Oracle Database Administration (DBA)',
    'Embedded Systems Engineering',
    'IT Support and Application Management',
    'Cybersecurity and Ethical Hacking',
    'Network Engineering and Infrastructure',
    'Full-Stack Web Development',
    'API Development and Integration',
    'Project Management Professional (PMP)',
    'Digital Forensics and Incident Response (DFIR)',
    'Technical Writing and Documentation',
    'Artificial Intelligence and Machine Learning',
    'Software Testing and Quality Assurance (QA)',
    'Business Analysis and Requirements Management',
    'Customer Service and Relationship Management',
    'Data Science and Advanced Analytics',
    'IT Support and Helpdesk Management',
    'Graphic Design and Visual Communication'
]

# دالة لتحويل الخبرة إلى قيم رقمية
def map_experience(value):
    mapping = {"Excellent": 3, "Average": 2, "Beginner": 1, "Not Interested": 0}
    return mapping.get(value, 0)

# واجهة Streamlit
def main():
    st.set_page_config(page_title="Career Path Recommender", page_icon="💼", layout="wide")
    
    # عرض صورة في البداية (اختياري مع التحقق من وجودها)
    if os.path.exists('career_image.jpg'):
        image = Image.open('career_image.jpg')
        st.image(image, use_column_width=True)
    
    st.title("Career Path Recommendation System")
    st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font">Please rate your skills/interest in the following areas:</p>', unsafe_allow_html=True)
    
    # نموذج الإدخال
    with st.form("career_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            db_fundamentals = st.selectbox("Database Fundamentals", ["Not Interested", "Beginner", "Average", "Excellent"])
            comp_arch = st.selectbox("Computer Architecture", ["Not Interested", "Beginner", "Average", "Excellent"])
            leadership = st.selectbox("Leadership Experience", ["Not Interested", "Beginner", "Average", "Excellent"])
            cybersecurity = st.selectbox("Cyber Security", ["Not Interested", "Beginner", "Average", "Excellent"])
            networking = st.selectbox("Networking", ["Not Interested", "Beginner", "Average", "Excellent"])
            soft_dev = st.selectbox("Software Development", ["Not Interested", "Beginner", "Average", "Excellent"])
            programming = st.selectbox("Programming Skills", ["Not Interested", "Beginner", "Average", "Excellent"])
            project_mgmt = st.selectbox("Project Management", ["Not Interested", "Beginner", "Average", "Excellent"])
            
        with col2:
            comp_forensics = st.selectbox("Computer Forensics Fundamentals", ["Not Interested", "Beginner", "Average", "Excellent"])
            tech_comm = st.selectbox("Technical Communication", ["Not Interested", "Beginner", "Average", "Excellent"])
            ai_ml = st.selectbox("AI/ML", ["Not Interested", "Beginner", "Average", "Excellent"])
            soft_eng = st.selectbox("Software Engineering", ["Not Interested", "Beginner", "Average", "Excellent"])
            business_analysis = st.selectbox("Business Analysis", ["Not Interested", "Beginner", "Average", "Excellent"])
            communication = st.selectbox("Communication Skills", ["Not Interested", "Beginner", "Average", "Excellent"])
            data_science = st.selectbox("Data Science", ["Not Interested", "Beginner", "Average", "Excellent"])
            troubleshooting = st.selectbox("Troubleshooting Skills", ["Not Interested", "Beginner", "Average", "Excellent"])
            graphics = st.selectbox("Graphics Designing", ["Not Interested", "Beginner", "Average", "Excellent"])
        
        submitted = st.form_submit_button("Get Career Recommendations")
    
    if submitted:
        try:
            # تحويل المدخلات لمصفوفة
            feature_array = np.array([[
                map_experience(db_fundamentals),
                map_experience(comp_arch),
                map_experience(leadership),
                map_experience(cybersecurity),
                map_experience(networking),
                map_experience(soft_dev),
                map_experience(programming),
                map_experience(project_mgmt),
                map_experience(comp_forensics),
                map_experience(tech_comm),
                map_experience(ai_ml),
                map_experience(soft_eng),
                map_experience(business_analysis),
                map_experience(communication),
                map_experience(data_science),
                map_experience(troubleshooting),
                map_experience(graphics)
            ]])
            
            # تطبيق التحجيم وPCA وFeature Selection
            scaled_features = scaler.transform(feature_array)
            pca_features = pca.transform(scaled_features)
            selected_features = selector.transform(pca_features)
            
            # التنبؤ
            probabilities = model.predict_proba(selected_features)
            top_classes_idx = np.argsort(-probabilities[0])[:3]
            
            st.success("Here are your top 3 recommended career paths:")
            
            for i, idx in enumerate(top_classes_idx):
                if 0 <= idx < len(class_names):
                    st.markdown(f"""
                    <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin:10px 0;">
                        <h3>#{i+1}: {class_names[idx]}</h3>
                        
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.text(traceback.format_exc())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Runtime error: {e}")
        st.text(traceback.format_exc())
