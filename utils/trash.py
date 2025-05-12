rf_full = 97.49
rf_dt = 96.7
xgb_full = 99.7
xgb_dt = 99.34


'''
with all features and all extraction
model_variant	Features	accuracy	cpu_time_s	energy_used_Wh
0	RF_FULL	234	0.994785	0.031250	0.000013
1	RF_DT	20	0.987831	0.046875	0.000013
2	XGB_FULL	234	0.998262	0.234375	0.000005
3	XGB_DT	20	0.998696	0.000000	0.000002


by removing zerocrossing, mad, skew, kertosis 
	model_variant	Features	accuracy	cpu_time_s	energy_used_Wh
0	RF_FULL	162	0.993481	0.093750	0.000020
1	RF_DT	20	0.983920	0.078125	0.000020
2	XGB_FULL	162	0.998696	0.203125	0.000004
3	XGB_DT	20	0.997392	0.250000	0.000002


by removing gps_alt
	model_variant	Features	accuracy	cpu_time_s	energy_used_Wh
0	RF_FULL	153	0.981312	0.109375	0.000019
1	RF_DT	10	0.930465	0.093750	0.000018
2	XGB_FULL	153	0.998262	0.250000	0.000005
3	XGB_DT	10	0.990439	0.234375	0.000003


'''
def main():
    st.title("Activity & Vehicle Mode Prediction")
    
    # File upload
    uploaded_file = st.file_uploader("Upload sensor data", type=["csv"])
    
    if uploaded_file is not None:
        # Load models
        models = load_models()

        # Preprocess only once and store in session_state
        if 'preprocessed' not in st.session_state:
            with st.spinner('Extracting Features...'):
                test_X, testX_dt, test_y = preprocess_data(uploaded_file)
                st.session_state.preprocessed = {
                    'test_X': test_X,
                    'testX_dt': testX_dt,
                    'test_y': test_y
                }

        if st.button('Predict'):
            with st.spinner('Making predictions...'):
                test_X = st.session_state.preprocessed['test_X']
                test_y = st.session_state.preprocessed['test_y']

                acc_rf_full = pr.predict_model(models['RF FULL'], test_X, test_y)
                acc_xgb_full = pr.predict_model(models['XGB FULL'], test_X, test_y)

            # Display results
            st.subheader("Predictions")
            results_df = pd.DataFrame([
                ['RF FULL', acc_rf_full],
                ['XGB FULL', acc_xgb_full]
            ], columns=['Model', 'Accuracy'])

            st.dataframe(results_df)
