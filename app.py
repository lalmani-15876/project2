import streamlit as st
import pandas as pd
import joblib
import utils.preprocessing as pr
import os
import utils.variables as cf

st.set_page_config(initial_sidebar_state="collapsed",layout="wide")

def load_models():
    le = joblib.load(os.path.join(cf.MODEL,cf.LABEL_ENCODER))
    models = {
        'RF FULL': joblib.load(os.path.join(cf.MODEL,cf.RF_FULL)),
        'RF DT': joblib.load(os.path.join(cf.MODEL,cf.RF_DT)),
        'RF ENTROPY': joblib.load(os.path.join(cf.MODEL,cf.RF_ENTROPY)),
        'RF CORR': joblib.load(os.path.join(cf.MODEL,cf.RF_CORR)),
        'XGB FULL': joblib.load(os.path.join(cf.MODEL,cf.XGB_FULL)),
        'XGB DT': joblib.load(os.path.join(cf.MODEL,cf.XGB_DT)),
        'XGB CORR': joblib.load(os.path.join(cf.MODEL,cf.XGB_CORR)),
        'XGB ENTROPY': joblib.load(os.path.join(cf.MODEL,cf.XGB_ENTROPY)),
    }
    return models,le

@st.cache_data
def preprocess_data(uploaded_file):
    """Process uploaded CSV file"""
    df = pr.laod_csv_data(uploaded_file)
    
    # Perform same preprocessing as training
    processed_df = pr.extract_features(df)

    le = joblib.load(os.path.join(cf.MODEL,cf.LABEL_ENCODER))
    test_X, testX_dt, testX_en, testX_corr, test_y = pr.remove_features(processed_df,le)
        
    return test_X, testX_dt, testX_en, testX_corr, test_y

def main():
    st.title("Activity & Vehicle Mode Prediction")
    
    # File upload
    uploaded_file = st.file_uploader("Upload sensor data", type=["csv"])
    
    if uploaded_file is not None:
        # Load models
        models,le = load_models()
        
        # Preprocess data
        with st.spinner('Extracting Features...'):
            test_X, testX_dt, testX_en, testX_corr, test_y = preprocess_data(uploaded_file)
            
        # Make predictions
        if st.button('Predict'):
            with st.spinner('Making predictions...'):
                acc_rf_full, rf_full_y = pr.predict_model(models['RF FULL'],test_X,test_y)
                acc_rf_dt, rf_dt_y = pr.predict_model(models['RF DT'],testX_dt,test_y)
                acc_rf_en, rf_en_y = pr.predict_model(models['RF ENTROPY'],testX_en,test_y)
                acc_rf_corr, rf_corr_y = pr.predict_model(models['RF CORR'],testX_corr,test_y)
                acc_xgb_full, xgb_full_y = pr.predict_model(models['XGB FULL'],test_X,test_y)
                acc_xgb_dt, xgb_dt_y = pr.predict_model(models['XGB DT'],testX_dt,test_y)
                acc_xgb_en, xgb_en_y = pr.predict_model(models['XGB ENTROPY'],testX_en,test_y)
                acc_xgb_corr, xgb_corr_y = pr.predict_model(models['XGB CORR'],testX_corr,test_y)

            with st.spinner('Generating Classification matrix...'):
                col1,col2 = st.columns(2)
                with col1:
                    st.pyplot(pr.plot_confusion_matrix(test_y,rf_full_y,"RF FULL",le))
                    st.pyplot(pr.plot_confusion_matrix(test_y,rf_dt_y,"RF DT",le))
                    st.pyplot(pr.plot_confusion_matrix(test_y,rf_en_y,"RF ENTROPY",le))
                    st.pyplot(pr.plot_confusion_matrix(test_y,rf_corr_y,"RF CORR",le))
                with col2:
                    st.pyplot(pr.plot_confusion_matrix(test_y,xgb_full_y,"XGB FULL",le))
                    st.pyplot(pr.plot_confusion_matrix(test_y,xgb_dt_y,"XGB DT",le))
                    st.pyplot(pr.plot_confusion_matrix(test_y,xgb_en_y,"XGB ENTROPY",le))
                    st.pyplot(pr.plot_confusion_matrix(test_y,xgb_corr_y,"XGB CORR",le))
                
            # Display results
            st.subheader("Predictions")
            results_df = pd.DataFrame([
                ['RF FULL', acc_rf_full],
                ['RF DT', acc_rf_dt],
                ['RF ENTROPY', acc_rf_en],
                # ['RF CORR', acc_rf_corr],
                ['XGB FULL', acc_xgb_full],
                # ['XGB DT', acc_xgb_dt],
                ['XGB ENTROPY', acc_xgb_en],
                ['XGB CORR', acc_xgb_corr]
            ], columns=['Model', 'Accuracy'])
            st.dataframe(results_df)

            with st.expander("See Classification Report"):
                col3,col4 = st.columns(2)
                with col3:
                    st.text(f'\nRF FULL Classification Report\n')
                    st.dataframe(pr.get_classification_report(test_y,rf_full_y,le))
                    st.text(f'\nRF DT Classification Report\n')
                    st.dataframe(pr.get_classification_report(test_y,rf_dt_y,le))
                    st.text(f'\nRF ENTROPY Classification Report\n')
                    st.dataframe(pr.get_classification_report(test_y,rf_en_y,le))
                    st.text(f'\nRF CORR Classification Report\n')
                    st.dataframe(pr.get_classification_report(test_y,rf_corr_y,le))
                with col4:
                    st.text(f'\nXGB FULL Classification Report\n')
                    st.dataframe(pr.get_classification_report(test_y,xgb_full_y,le))
                    st.text(f'\nXGB DT Classification Report\n')
                    st.dataframe(pr.get_classification_report(test_y,xgb_dt_y,le))
                    st.text(f'\nXGB ENTROPY Classification Report\n')
                    st.dataframe(pr.get_classification_report(test_y,xgb_en_y,le))
                    st.text(f'\nXGB CORR Classification Report\n')
                    st.dataframe(pr.get_classification_report(test_y,xgb_corr_y,le))
                
if __name__ == "__main__":
    main()