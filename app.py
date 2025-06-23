import streamlit as st
import joblib
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import base64
from gtts import gTTS
import torch
import pickle
import json
import random
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import traceback

# Set page config
st.set_page_config(page_title="Data Explainability Chatbot", layout="wide")

# =============================================
# Common Functions
# =============================================

def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    with open("response.mp3", "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()
    return f'<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'

# =============================================
# Interface 1 - ML Model 
# =============================================

def load_ml_components():
    try:
        model = joblib.load("chatbot_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        explainer = joblib.load("shap_explainer.pkl")
        return model, vectorizer, label_encoder, explainer
    except Exception as e:
        st.error(f"Error loading ML components: {e}")
        return None, None, None, None

def ml_chatbot_response(user_question, model, vectorizer, label_encoder, explainer):
    try:
        user_vector = vectorizer.transform([user_question])
        predicted_proba = model.predict_proba(user_vector)
        predicted_label = model.predict(user_vector)[0]
        confidence = np.max(predicted_proba) * 100

        if confidence < 50:
            return "⚠️ Sorry, I am not trained to answer this question. Please try a different one.", None, None, confidence

        predicted_label = int(predicted_label)
        predicted_answer = label_encoder.inverse_transform([predicted_label])[0]
        user_vector_dense = user_vector.toarray()
        shap_values = explainer(user_vector_dense)

        return predicted_answer, shap_values, user_vector_dense, confidence
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "⚠️ An error occurred while processing your request.", None, None, 0

def show_ml_interface():
    st.title("🤖 ML Chatbot with Explainability (SHAP & LIME)")

    st.sidebar.title("📌 About This Chatbot")
    st.sidebar.markdown("""
This chatbot is powered by a **Machine Learning (ML) model** trained to classify user queries and provide relevant responses.

#### 🧠 **Model Used:**
- The chatbot uses a **RandomForestClassifier**.
- This model is efficient for **text classification** tasks.

#### 🔍 **How the Text is Processed?**
1. **TF-IDF Vectorization**  
   - Converts user input into a numerical format.
   - Gives more importance to **rare but meaningful words**.
  
2. **ML Model Prediction**  
   - The trained model predicts the most likely response category.
   - The label is converted back into human-readable text.

#### 🔥 **Explainability with SHAP & LIME**
- **SHAP (SHapley Additive exPlanations)**: Explains which words influenced the chatbot's response the most.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Provides local feature importance for specific predictions.
""")
    
    # Load ML components
    model, vectorizer, label_encoder, explainer = load_ml_components()
    if None in [model, vectorizer, label_encoder, explainer]:
        st.error("Failed to load ML components. Please check your model files.")
        return
    
    # Initialize chat history
    if "ml_chat_history" not in st.session_state:
        st.session_state.ml_chat_history = []

    # User input
    if user_input := st.chat_input("Type your message..."):
        response, shap_values, user_vector_dense, confidence = ml_chatbot_response(
            user_input, model, vectorizer, label_encoder, explainer
        )

        # Update chat history
        st.session_state.ml_chat_history.append({"role": "user", "message": user_input})
        st.session_state.ml_chat_history.append({"role": "bot", "message": response})

        # Display chat history
        st.markdown("### 📜 Chat History")
        for chat in st.session_state.ml_chat_history:
            if chat["role"] == "user":
                st.markdown(f"**You:** {chat['message']}")
            else:
                st.markdown(f"**🤖 Bot:** {chat['message']}")

        # Confidence level
        st.markdown(f"🔹 **Confidence Level:** {confidence:.2f}%")
        st.progress(confidence / 100)

        # Text-to-speech
        st.markdown(text_to_speech(response), unsafe_allow_html=True)

        # Only show explanations if we have a valid response with good confidence
        if confidence >= 50 and shap_values is not None:
            # SHAP Explanation
            st.markdown("### 🔍 SHAP Explanation")
            shap_values_2d = shap_values.values[0]
            feature_names = vectorizer.get_feature_names_out()[:shap_values_2d.shape[1]]
            shap_df = pd.DataFrame(shap_values_2d, columns=feature_names)
            shap_df["Total SHAP Impact"] = np.abs(shap_df).sum(axis=1)
            shap_df_sorted = shap_df[["Total SHAP Impact"]].sort_values(by="Total SHAP Impact", ascending=False)
            st.bar_chart(shap_df_sorted)

            top_shap_words = shap_df_sorted.index[:3].tolist()
            st.markdown(f"#### ✅ Conclusion:\nThis response was strongly influenced by words like '{top_shap_words[0]}', '{top_shap_words[1]}', and '{top_shap_words[2]}'.")

            # LIME Explanation
            # LIME Explanation
            try:
                from lime.lime_text import LimeTextExplainer
                
                # Create LIME Text Explainer
                lime_explainer = LimeTextExplainer(
                    class_names=label_encoder.classes_,
                    split_expression=lambda x: x.split(),
                    bow=False
                )
                
                # Predict function for LIME
                def predict_proba(texts):
                    text_vector = vectorizer.transform(texts)
                    return model.predict_proba(text_vector)
                
                # Generate explanation
                exp = lime_explainer.explain_instance(
                    user_input,
                    predict_proba,
                    num_features=5,
                    top_labels=1
                )
                
                # Display LIME explanation in the requested format
                st.markdown("### 🔍 LIME Explanation (Words that influenced prediction):")
                lime_list = exp.as_list(label=exp.available_labels()[0])
                
                # Display each feature with its contribution
                for feature, weight in lime_list[:3]:  # Show top 3 features
                    st.markdown(f"🔹 {feature} → Contribution: {abs(weight):.4f}")
                
                # Optional visualization
                with st.expander("View detailed explanation"):
                    lime_df = pd.DataFrame(lime_list, columns=["Feature", "Importance"])
                    st.bar_chart(lime_df.set_index("Feature"))
                
            except Exception as e:
                st.warning(f"Could not generate LIME explanation: {str(e)}")

    if st.button("← Back to Model Selection"):
           st.session_state.current_interface = "selection"
           st.rerun()

# =============================================
# Interface 2 - DL Model (Neural Network)
# =============================================

# Neural Network Model
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout1(self.bn1(self.relu1(self.fc1(x))))
        x = self.dropout2(self.bn2(self.relu2(self.fc2(x))))
        return self.softmax(self.fc3(x))

@st.cache_resource
def load_dl_model():
    try:
        with open("words.pkl", "rb") as f:
            words = pickle.load(f)
        with open("classes.pkl", "rb") as f:
            classes = pickle.load(f)
        with open("intents.json", "r") as f:
            intents = json.load(f)
            
        input_size = len(words)
        hidden_size = 8
        output_size = len(classes)
        
        model = NeuralNetwork(input_size, hidden_size, output_size)
        model.load_state_dict(torch.load("final_model.pth", map_location=torch.device("cpu")))
        model.eval()
        
        return model, words, classes, intents
    except Exception as e:
        st.error(f"Error loading DL model: {e}")
        return None, None, None, None

def preprocess_sentence(sentence, words):
    return [word for word in sentence.lower().split() if word in words]

def sentence_to_features(sentence_words, words):
    return torch.tensor(
        [1 if word in sentence_words else 0 for word in words], dtype=torch.float32
    ).unsqueeze(0)


def show_dl_interface():
    st.title("🤖 DL Chatbot with Explainability (SHAP & LIME)")

    st.sidebar.title("📌 About This Chatbot")
    st.sidebar.markdown("""
This chatbot is powered by a **Deep Learning (DL) model** trained to classify user queries and provide relevant responses.

#### 🧠 **Model Used:**
- The chatbot uses a **Neural Network** with word embeddings.
- This model is effective for **natural language understanding** tasks.

#### 🔍 **How the Text is Processed?**
1. **Tokenization & Embeddings**  
   - Converts user input into a format suitable for deep learning models.
   - Uses **bag-of-words representation**.

2. **DL Model Prediction**  
   - The trained model predicts the most likely response category.
   - The output is decoded into a human-readable response.

#### 🔥 **Explainability with SHAP & LIME**
- **SHAP (SHapley Additive exPlanations)**: Explains which words influenced the chatbot's response the most.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Provides local feature importance for specific predictions.
""")
    
    # Load DL components
    model, words, classes, intents = load_dl_model()
    if None in [model, words, classes, intents]:
        st.error("Failed to load DL components. Please check your model files.")
        return
    
    # Initialize chat history
    if "dl_chat_history" not in st.session_state:
        st.session_state.dl_chat_history = []

    # User input
    if user_input := st.chat_input("Type your message..."):
        sentence_words = preprocess_sentence(user_input, words)
        show_explanations = True
        
        if not sentence_words:
            response = "⚠️ I didn't understand that. Can you rephrase with different words?"
            show_explanations = False
        else:
            features = sentence_to_features(sentence_words, words)
            with torch.no_grad():
                outputs = model(features)
                confidence, predicted_class = torch.max(outputs, dim=1)
            
            if confidence.item() < 0.3:
                response = "⚠️ I'm not sure about this. Can you clarify or ask something else?"
                show_explanations = False
            else:
                predicted_tag = classes[predicted_class.item()]
                intent = next((intent for intent in intents["intents"] if intent["tag"] == predicted_tag), None)
                if intent:
                    response = random.choice(intent["responses"])
                else:
                    response = "⚠️ I don't understand this question."
                    show_explanations = False
        
        # Update chat history
        st.session_state.dl_chat_history.append({"role": "user", "message": user_input})
        st.session_state.dl_chat_history.append({"role": "bot", "message": response})

        # Display chat history
        st.markdown("### 📜 Chat History")
        for chat in st.session_state.dl_chat_history:
            if chat["role"] == "user":
                st.markdown(f"**You:** {chat['message']}")
            else:
                st.markdown(f"**🤖 Bot:** {chat['message']}")

        # Confidence level
        if show_explanations:
            st.markdown(f"🔹 **Confidence Level:** {confidence.item()*100:.2f}%")
            st.progress(confidence.item())

        # Text-to-speech
        st.markdown(text_to_speech(response), unsafe_allow_html=True)

        # Only show explanations if we have a valid response
        if show_explanations:
            # LIME Explanation
            try:
                st.markdown("### 🔍 LIME Explanation")
                
                def predict_fn(x):
                    x_tensor = torch.tensor(x, dtype=torch.float32)
                    with torch.no_grad():
                        return model(x_tensor).numpy()
                
                lime_explainer = LimeTabularExplainer(
                    training_data=np.zeros((1, len(words))),  # Dummy data
                    feature_names=words,
                    class_names=classes,
                    mode="classification"
                )
                
                exp = lime_explainer.explain_instance(
                    sentence_to_features(sentence_words, words).numpy()[0], 
                    predict_fn, 
                    num_features=10
                )
                
                # Display LIME explanation
                lime_list = exp.as_list()
                
                
                # Detailed visualization
                with st.expander("📊 View Detailed LIME Analysis"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        lime_df = pd.DataFrame(lime_list, columns=["Feature", "Importance"])
                        lime_df = lime_df.sort_values("Importance", ascending=False).head(10)
                        colors = ['#4CAF50' if x > 0 else '#F44336' for x in lime_df['Importance']]
                        ax.barh(lime_df['Feature'], lime_df['Importance'], color=colors)
                        ax.set_xlabel('Feature Importance')
                        ax.set_title('Top Features by LIME Importance')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with col2:
                        st.markdown("**Raw Data Values**")
                        st.dataframe(
                            lime_df.sort_values("Importance", ascending=False),
                            height=300,
                            use_container_width=True
                        )
                    
            except Exception as e:
                st.warning(f"Could not generate LIME explanation: {str(e)}")

            # SHAP Explanation (updated with fix and improvements)
            try:
                st.markdown("### 🔍 SHAP Explanation")
                
                # Convert input to proper format
                input_features = sentence_to_features(sentence_words, words).numpy()[0:1]
                
                # Define prediction function compatible with SHAP
                def predict_fn(x):
                    x_tensor = torch.tensor(x, dtype=torch.float32)
                    with torch.no_grad():
                        return model(x_tensor).numpy()
                
                # Create explainer with appropriate background
                background = np.zeros((1, len(words)))
                explainer = shap.Explainer(
                    predict_fn, 
                    background,
                    feature_names=words
                )
                
                # Calculate SHAP values
                shap_values = explainer(input_features)
                predicted_class_idx = predicted_class.item()
                
                # Create DataFrame for visualization
                shap_df = pd.DataFrame({
                    'Feature': words,
                    'SHAP Value': shap_values.values[0, :, predicted_class_idx],
                    'Present': input_features[0].astype(bool)
                }).sort_values('SHAP Value', key=abs, ascending=False)
                
                # Main visualization in expander
                with st.expander("📊 View Detailed SHAP Analysis"):
                    tab1, tab2 = st.tabs(["Visualization", "Raw Data"])
                    
                    with tab1:
                        
                        # Waterfall plot
                        st.markdown("**Waterfall Plot**")
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        shap.plots.waterfall(
                            shap_values[0, :, predicted_class_idx],
                            max_display=15,
                            show=False
                        )
                        plt.tight_layout()
                        st.pyplot(fig2)
                        plt.close(fig2)
                    
                    with tab2:
                        st.markdown("**All Features Impact**")
                        st.dataframe(
                            shap_df.style.format({'SHAP Value': '{:.4f}'})
                            .apply(lambda x: ['background-color: #E8F5E9' if x['Present'] and x['SHAP Value'] > 0 
                                         else 'background-color: #FFEBEE' if x['Present'] and x['SHAP Value'] < 0 
                                         else '' for i in x], axis=1),
                            height=500,
                            use_container_width=True
                        )
                
                # Decision plot in separate expander
                with st.expander("📈 View SHAP Decision Plot"):
                    fig3, ax3 = plt.subplots(figsize=(10, 8))
                    shap.decision_plot(
                        shap_values.base_values[0, predicted_class_idx],
                        shap_values.values[0, :, predicted_class_idx],
                        feature_names=words,
                        show=False
                    )
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close(fig3)
                
                # Impact statistics
                st.markdown("#### 📝 Impact Statistics")
                pos_impact = shap_df[shap_df['SHAP Value'] > 0]['SHAP Value'].sum()
                neg_impact = shap_df[shap_df['SHAP Value'] < 0]['SHAP Value'].sum()
                net_impact = pos_impact + neg_impact
                
                cols = st.columns(3)
                cols[0].metric("Positive Impact", f"{pos_impact:.3f}")
                cols[1].metric("Negative Impact", f"{neg_impact:.3f}")
                cols[2].metric("Net Impact", f"{net_impact:.3f}")
                
            except Exception as e:
                st.warning(f"Could not generate SHAP explanation: {str(e)}")
                if st.session_state.get('debug_mode', False):
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")

    if st.button("← Back to Model Selection"):
        st.session_state.current_interface = "selection"
        st.rerun()

# =============================================
# Model Selection Interface
# =============================================

def show_selection_interface():
    st.title("🤖 Data Explainability Chatbot")
    st.markdown("### Select a Model Type to Chat With")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='border: 1px solid #e1e4e8; border-radius: 10px; padding: 20px; text-align: center; height: 300px;'>
            <h3>Machine Learning Model</h3>
            <p>RandomForestClassifier with TF-IDF vectorization</p>
            <p>Good for simpler text classification tasks</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select ML Model", key="ml_button"):
            st.session_state.current_interface = "ml"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style='border: 1px solid #e1e4e8; border-radius: 10px; padding: 20px; text-align: center; height: 300px;'>
            <h3>Deep Learning Model</h3>
            <p>Neural Network with word embeddings</p>
            <p>Better for complex language understanding</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select DL Model", key="dl_button"):
            st.session_state.current_interface = "dl"
            st.rerun()

    st.markdown("---")
    st.markdown("""
    ### About This Application
    This application demonstrates two different approaches to building chatbots:
    - **ML Model**: Traditional machine learning approach using Naive Bayes Classifier
    - **DL Model**: Deep learning approach using a neural network
    
    Both interfaces include explainability features using SHAP and LIME to help understand how the models make decisions.
    """)

# =============================================
# Main App Logic
# =============================================

if "current_interface" not in st.session_state:
    st.session_state.current_interface = "selection"

if st.session_state.current_interface == "selection":
    show_selection_interface()
elif st.session_state.current_interface == "ml":
    show_ml_interface()
elif st.session_state.current_interface == "dl":
    show_dl_interface()