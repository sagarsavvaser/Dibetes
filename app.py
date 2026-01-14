import streamlit as st
import pickle
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "logistic_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
