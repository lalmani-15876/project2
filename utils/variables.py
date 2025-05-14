
# Configuration
MODEL = "models"
RF_FULL = "RF_FULL.pkl"
RF_DT = "RF_DT.pkl"
XGB_FULL = "XGB_FULL.pkl"
XGB_DT = "XGB_DT.pkl"
LABEL_ENCODER = "label_encoder.pkl"

sensor_features = [
    'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z',
    'Gyroscope_X', 'Gyroscope_Y', 'Gyroscope_Z',
    'Magnetometer_X', 'Magnetometer_Y', 'Magnetometer_Z',
    'GPS_Speed','Sound Level', 'Activity'
]

# top_dt_cols = ['GPS_Speed_min', 'Gyroscope_Z_iqr', 'Sound Level_rms',
#        'Accelerometer_Z_min', 'Sound Level_median', 'Gyroscope_X_iqr',
#        'GPS_Speed_max', 'Magnetometer_Y_min', 'Magnetometer_Z_min',
#        'Sound Level_max']

top_dt_col = ['GPS_Speed_min', 'Gyroscope_Z_iqr', 'Sound Level_rms', 'GPS_Speed_rms',
       'Accelerometer_Z_min', 'Accelerometer_Y_mean', 'Accelerometer_Y_max',
       'Magnetometer_Z_min', 'Magnetometer_Y_min', 'Sound Level_max']

if __name__ == "__main__":
    pass