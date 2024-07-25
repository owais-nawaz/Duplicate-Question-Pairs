import streamlit as st
import helper
import helper_bilstm
import pickle
import tensorflow as tf
import numpy as np


# Load models
@st.cache_resource
def load_et_model():
    return pickle.load(open("./streamlit-app/model_et.pkl", "rb"))


@st.cache_resource
def load_rf_model():
    return pickle.load(open("./streamlit-app/rf_model.pkl", "rb"))


@st.cache_resource
def load_xgb_model():
    return pickle.load(open("./streamlit-app/xgb_model.pkl", "rb"))


@st.cache_resource
def load_bilstm_model():
    try:
        return tf.keras.models.load_model("./streamlit-app/final_lstm_model.h5")
    except Exception as e:
        st.error(f"Error loading BiLSTM model: {e}")
        return None


def main():
    st.title("Question Pair Similarity")

    st.sidebar.title("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose a model",
        (
            "BiLSTM",
            "Extra Trees Classifier",
            "Random Forest Classifier",
            "XGB Classifier",
        ),
    )

    st.header("Check For Duplicate Question Pairs")

    q1 = st.text_input("Enter question 1")
    q2 = st.text_input("Enter question 2")

    if st.button("Check"):
        if not q1 or not q2:
            st.error("Both questions must be provided.")
            return

        result = None
        if model_choice == "Extra Trees Classifier":
            et_model = load_et_model()
            try:
                query = helper.query_point_creator(q1, q2)
                result = et_model.predict(query)[0]
            except Exception as e:
                st.error(f"Error making prediction with Extra Trees model: {e}")
        elif model_choice == "Random Forest Classifier":
            rf_model = load_rf_model()
            try:
                query = helper.query_point_creator(q1, q2)
                result = rf_model.predict(query)[0]
            except Exception as e:
                st.error(f"Error making prediction with Random Forest model: {e}")
        elif model_choice == "XGB Classifier":
            xgb_model = load_xgb_model()
            try:
                query = helper.query_point_creator(q1, q2)
                result = xgb_model.predict(query)[0]
            except Exception as e:
                st.error(f"Error making prediction with XGB Classifier model: {e}")
        elif model_choice == "BiLSTM":
            bilstm_model = load_bilstm_model()
            if bilstm_model is not None:
                try:
                    q1_padded, q2_padded, len_features_test = (
                        helper_bilstm.bilstm_features(q1, q2)
                    )
                    result = bilstm_model.predict(
                        [q1_padded, q2_padded, len_features_test]
                    )[0][0]
                except Exception as e:
                    st.error(f"Error making prediction with BiLSTM model: {e}")
        if result is not None:
            if result > 0.5:
                st.header("Duplicate Questions")
            else:
                st.header("Non-Duplicate Questions")
        else:
            st.error("No result to display. Please check the input and try again.")


if __name__ == "__main__":
    main()
