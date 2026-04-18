import streamlit as st
import pickle
import numpy as np

# load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

feature_names = vectorizer.get_feature_names_out()

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI Powered News Verification System 🚀</h4>", unsafe_allow_html=True)

news = st.text_area("📝 Enter your news here:", height=150)

if st.button("🔍 Check News"):

    # ✅ prediction
    news_vec = vectorizer.transform([news])
    prediction = model.predict(news_vec)
    prob = model.predict_proba(news_vec)

    # ✅ important words
    indices = news_vec.toarray()[0].argsort()[-5:]
    words = [feature_names[i] for i in indices]

    # ✅ reason
    if prediction[0] == 1:
        reason = "This news appears real because it uses words like: " + ", ".join(words)
    else:
        reason = "This news seems fake because it uses suspicious words like: " + ", ".join(words)

    # ✅ confidence (MOVE HERE 🔥)
    confidence = prob[0][1] if prediction[0] == 1 else prob[0][0]

    st.progress(float(confidence))
    st.write(f"🔎 Confidence Score: {confidence*100:.2f}%")

# ✅ RESULT UI
    if prediction[0] == 1:
        st.markdown(f"""
        <div style='background-color:#d4edda;padding:15px;border-radius:10px'>
        <h3>✅ Real News</h3>
        <p>Confidence: {prob[0][1]*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background-color:#f8d7da;padding:15px;border-radius:10px'>
        <h3>❌ Fake News</h3>
        <p>Confidence: {prob[0][0]*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

# ✅ WARNING (ADD HERE 🔥)
    if confidence < 0.6:
        st.warning("⚠️ This prediction may not be very accurate. Please verify manually.")