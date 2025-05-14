import pandas as pd
import streamlit as st
from utils.preprocessing import laod_csv_data,extract_features


def check_duplicate_data(train,test):
    train_tuples = [tuple(row) for row in train.values]
    test_tuples = [tuple(row) for row in test.values]
    
    # Find intersection
    common = set(train_tuples) & set(test_tuples)
    return len(common)

def main():
    Test_file = st.file_uploader("Upload Test data", type=["csv"])
    if Test_file is not None:
        test = laod_csv_data(Test_file)
    
    Train_file = st.file_uploader("Upload Train data", type=["csv"])
    if Train_file is not None:
        train = laod_csv_data(Train_file)

    if st.button('Check Duplicate Rows'):    
        # st.text(f"Total common rows Before Feature Extraction: {check_duplicate_data(train,test)}")
        test = extract_features(test)
        train = extract_features(train)

        st.text(f"Total common rows After Feature Extraction: {check_duplicate_data(train,test)}")

if __name__ == "__main__":
    main()