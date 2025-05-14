import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import utils.variables as variables

def laod_csv_data(file):
    df = pd.read_csv(file)
    return df[variables.sensor_features]

def extract_features(df, window_size=120, step_size=30):
    label_col = 'Activity'
    sensor_cols = [col for col in df.columns if col != label_col]

    features = []
    for start_idx in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start_idx : start_idx + window_size]
        window_feats = {}

        # Calculate stats for each sensor column
        for col in sensor_cols:
            data = window[col].values
            window_feats.update({
                f"{col}_mean": np.mean(data),
                f"{col}_std": np.std(data),
                f"{col}_max": np.max(data),
                f"{col}_min": np.min(data),
                f"{col}_median": np.median(data),
                f"{col}_rms" : np.sqrt(np.mean(data**2)),
                f"{col}_range": np.max(data) - np.min(data),
                f"{col}_iqr": np.percentile(data, 75) - np.percentile(data, 25),
            })
        window_feats[label_col] = window[label_col].iloc[-1]
        features.append(window_feats)

    return pd.DataFrame(features)


def remove_features(test_df,le):
    test_X = test_df.drop('Activity', axis=1)
    test_y = test_df['Activity']

    testX_dt = test_X[variables.top_dt_col]
    
    test_y = le.transform(test_y)

    return test_X, testX_dt, test_y

def plot_confusion_matrix(y_true, y_pred, tag, label_encoder):
    class_names = label_encoder.inverse_transform(np.unique(y_pred))
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names,
                ax=ax)
    ax.set_title(f'{tag} Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Use tight layout
    plt.tight_layout()
    
    return fig


def get_classification_report(y_true, y_pred, label_encoder):
    
    target_names = label_encoder.inverse_transform(np.unique(y_pred))
    report_dict = classification_report(
        y_true, 
        y_pred, 
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    return pd.DataFrame(report_dict).transpose()

def predict_model(model,X,y):
    pred_y = model.predict(X)
    acc_score = accuracy_score(y, pred_y)
    return acc_score, pred_y

if __name__ == "__main__":
    pass