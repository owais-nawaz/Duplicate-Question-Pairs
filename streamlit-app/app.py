import streamlit as st
import helper
import pickle

et_model = pickle.load(open("model_et.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))

st.header("Check For Duplicate Question Pairs")

q1 = st.text_input("Enter question 1")
q2 = st.text_input("Enter question 2")

if st.button("Check"):
    query = helper.query_point_creator(q1, q2)
    result = et_model.predict(query)[0]

    if result:
        st.header("Duplicate Questions")
    else:
        st.header("Non-Duplicate Questions")
