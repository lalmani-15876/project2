
# Configuration
MODEL = "models"
RF_FULL = "RF_FULL.pkl"
RF_DT = "RF_DT.pkl"
RF_CORR = "RF_CORR.pkl"
RF_ENTROPY = "RF_ENTROPY.pkl"
XGB_FULL = "XGB_FULL.pkl"
XGB_DT = "XGB_DT.pkl"
XGB_ENTROPY = "XGB_ENTROPY.pkl"
XGB_CORR = "XGB_CORR.pkl"
LABEL_ENCODER = "label_encoder.pkl"

sensor_features = [
    'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z',
    'Gyroscope_X', 'Gyroscope_Y', 'Gyroscope_Z',
    'Magnetometer_X', 'Magnetometer_Y', 'Magnetometer_Z',
    'GPS_Speed','Sound Level', 'Activity'
]

top_entropy_col = ['GPS_Speed_iqr', 'GPS_Speed_std', 'Sound Level_iqr', 'GPS_Speed_range',
       'Magnetometer_Y_iqr', 'Magnetometer_Z_iqr', 'Sound Level_std',
       'Magnetometer_X_iqr', 'Magnetometer_Y_std', 'Magnetometer_Z_std',
       'Gyroscope_Y_median', 'Magnetometer_X_std', 'Gyroscope_Z_median',
       'Gyroscope_Y_mean', 'Accelerometer_Y_iqr', 'Magnetometer_Z_range',
       'Sound Level_range', 'Magnetometer_Y_range', 'Accelerometer_Z_iqr',
       'Gyroscope_X_mean']

top_dt_col = ['GPS_Speed_min', 'Gyroscope_Z_iqr', 'Sound Level_rms', 'GPS_Speed_rms',
       'Accelerometer_Z_min', 'Accelerometer_Y_mean', 'Accelerometer_Y_max',
       'Magnetometer_Z_min', 'Magnetometer_Y_min', 'Sound Level_max',
       'Magnetometer_X_mean', 'Sound Level_iqr', 'Gyroscope_X_iqr',
       'Magnetometer_Z_range', 'Sound Level_mean', 'Magnetometer_Z_iqr',
       'Accelerometer_Z_mean', 'Gyroscope_Z_mean', 'Magnetometer_X_min',
       'Sound Level_min']

top_corr_col = ['Sound Level_rms', 'Sound Level_median', 'Sound Level_mean',
       'GPS_Speed_min', 'GPS_Speed_rms', 'GPS_Speed_mean', 'GPS_Speed_median',
       'GPS_Speed_max', 'Sound Level_max', 'Accelerometer_Y_max',
       'Sound Level_min', 'Accelerometer_Y_mean', 'Accelerometer_X_max',
       'Accelerometer_Y_median', 'Accelerometer_Z_range',
       'Accelerometer_Z_std', 'Accelerometer_Z_min', 'Accelerometer_Y_rms',
       'Accelerometer_Y_range', 'Magnetometer_Z_max']

if __name__ == "__main__":
    pass