import streamlit as st
from sklearn.model_selection import train_test_split
import utils.preprocessing as pr
import utils.project_2 as p2
import utils.variables as cf

from joblib import dump

def split_train_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,  # for reproducibility
        shuffle=True,     # shuffling the data
        stratify=y       # maintain class distribution
    )
    # Verify shapes
    # st.text(f"Train: {X_train.shape}, {y_train.shape}")
    # st.text(f"Test: {X_test.shape}, {y_test.shape}")
    return X_train, X_test, y_train, y_test

@st.cache_data
def preprocess_data(train_df):
    agg_df = pr.extract_features(train_df)
    X,y,le = p2.split_X_y(agg_df)
    return X,y,le

def top_dt_features(X,y):
    top_10_features = p2.top_10features_from_DT(X,y)
    st.text(top_10_features)
    cf.top_dt_col = top_10_features
    X_dt = X[top_10_features]
    X_train_dt, X_test_dt, y_train_dt, y_test_dt = split_train_test(X_dt,y)
    return X_train_dt, X_test_dt, y_train_dt, y_test_dt

@st.cache_resource
def model_details(_model,X_train, y_train,X_test,y_test,tag,_le):
    p2.log_model(_model,X_train, y_train,X_test,y_test,tag,_le)
    # st.pyplot(p2.visualize_shap_per_class(_model, X_train, _le))

def main():
    st.header("Training of Models for Vechical and Activity Mode Detection")

    # File upload
    uploaded_file = st.file_uploader("Upload sensor data", type=["csv"])
    
    if uploaded_file is not None:
        train_df = pr.laod_csv_data(uploaded_file)

        st.dataframe(train_df.head(5))
        # st.pyplot(p2.plot_activity_counts(train_df))
        if st.button("Train Model"):
            X,y,le = preprocess_data(train_df)
            
            X_train, X_test, y_train, y_test = split_train_test(X,y)
            
            X_train_dt, X_test_dt, y_train_dt, y_test_dt = top_dt_features(X,y)

            rf_model,xgb_model = p2.initialize_models()

            rf_full = rf_model.fit(X_train, y_train)
            dump(rf_full, 'models/temp/RF_FULL.pkl')
            with st.expander("See Details of RF FULL"):
                model_details(rf_full,X_train, y_train,X_test,y_test,"RF FULL",le)
                st.image('models/cache/rf_full_class.png')

            rf_dt = rf_model.fit(X_train_dt,y_train_dt)
            dump(rf_dt,'models/temp/RF_DT.pkl')
            with st.expander("See Details of RF DT"):
                model_details(rf_dt,X_train_dt,y_train_dt,X_test_dt,y_test_dt,"RF_DT",le)
                st.image('models/cache/rf_dt_class.png')

            xgb_full = xgb_model.fit(X_train, y_train)
            dump(xgb_full,'models/temp/XGB_FULL.pkl')
            with st.expander("See Details of XGB FULL"):
                model_details(xgb_full,X_train, y_train,X_test,y_test,"XGB_FULL",le)
                st.image('models/cache/xgb_full_class.png')

            xgb_dt = xgb_model.fit(X_train_dt,y_train_dt)
            dump(xgb_dt,'models/temp/XGB_DT.pkl')
            with st.expander("See Details of XGB DT"):
                model_details(xgb_dt,X_train_dt,y_train_dt,X_test_dt,y_test_dt,"XGB_DT",le)
                st.image('models/cache/xgb_dt_class.png')

            variant = {
                'FULL': (X_train, y_train,X_test,y_test),
                'DT': (X_train_dt,y_train_dt,X_test_dt,y_test_dt)
            }
            models = {
                'RF': rf_model,
                'XGB': xgb_model
            }
            st.dataframe(p2.get_metrics(models,variant))

if __name__ == "__main__":
    main()