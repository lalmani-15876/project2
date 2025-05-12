import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load, dump
import shap
import time, psutil
import utils.preprocessing as pr
import streamlit as st

def plot_activity_counts(df, activity_column='Activity'):

    activity_counts = df[activity_column].value_counts()

    fig, ax = plt.subplots()
    activity_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel(activity_column)
    ax.set_ylabel("Count")
    ax.set_title(f"Value Counts of '{activity_column}'")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def split_X_y(agg_df):
    X = agg_df.drop('Activity', axis=1)
    y = agg_df['Activity']
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    dump(le, 'models/temp/label_encoder.pkl')
    return X,y,le

def top_10features_from_DT(X,y):
    dt_model = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt_model.fit(X, y)
    dt_series = pd.Series(dt_model.feature_importances_, index=X.columns)
    top_dt_cols = dt_series.sort_values(ascending=False).head(10)
    return top_dt_cols.index


def initialize_models():
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        objective='multi:softmax',
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )

    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=9,
        max_features='sqrt',
        min_samples_split=2,
        max_leaf_nodes=None,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    return rf_model,xgb_model


def log_model(model,X_train, y_train,X_test,y_test,tag,label_encoder):
    st.text(f"\n=== {tag} ===")
    y_pred = model.predict(X_test)
    st.write(f"Accuracy: {accuracy_score(y_test,y_pred):.4f}")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    st.write(f"5-fold CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    col1,col2 = st.columns(2)
    with col1:
        st.text("\nClassification Report:")
        st.dataframe(pr.get_classification_report(y_test,y_pred,label_encoder))
    with col2:
        st.text("\nConfusion Matrix")
        st.pyplot(pr.plot_confusion_matrix(y_test,y_pred,tag,label_encoder))

def evaluate_model(model,X_train, y_train,X_test,  y_test,
                   tag,
                   P_idle=5.0,    # watts, your CPU’s idle power
                   P_peak=65.0):   # watts, your CPU’s max power
    """
    Trains `model` on (X_train, y_train), evaluates on (X_test, y_test),
    and returns timing, CPU usage, estimated energy, and accuracy.
    Energy is estimated as:
        P_avg = P_idle + cpu_util * (P_peak - P_idle)
        E_Wh  = P_avg * (wall_time_s / 3600)
    """
    proc = psutil.Process()

    # ----- TRAINING MEASUREMENT -----
    start_wall = time.perf_counter()
    cpu_start = proc.cpu_times()           # user+system at start

    model.fit(X_train, y_train)

    train_wall = time.perf_counter() - start_wall
    cpu_end = proc.cpu_times()
    train_cpu = ((cpu_end.user + cpu_end.system)
                 - (cpu_start.user + cpu_start.system))

    # Avoid division by zero
    train_util = train_cpu / train_wall if train_wall > 0 else 0.0
    P_train = P_idle + train_util * (P_peak - P_idle)
    E_train_Wh = P_train * (train_wall / 3600)

    # ----- PREDICTION MEASUREMENT -----
    start_wall = time.perf_counter()
    cpu_start = proc.cpu_times()

    y_pred = model.predict(X_test)

    test_wall = time.perf_counter() - start_wall
    cpu_end = proc.cpu_times()
    test_cpu = ((cpu_end.user + cpu_end.system)
                - (cpu_start.user + cpu_start.system))

    test_util = test_cpu / test_wall if test_wall > 0 else 0.0
    P_test = P_idle + test_util * (P_peak - P_idle)
    E_test_Wh = P_test * (test_wall / 3600)

    # ----- METRICS -----
    accuracy = accuracy_score(y_test, y_pred)

    results = {
        'model_variant': tag.split(':')[0],
        'n_features':    X_train.shape[1],
        'train_time_s':  train_wall,
        'train_cpu_s':   train_cpu,
        'train_energy_Wh': E_train_Wh,
        'test_time_s':   test_wall,
        'test_cpu_s':    test_cpu,
        'test_energy_Wh':  E_test_Wh,
        'accuracy':      accuracy
    }
    return results


def visualize_shap_per_class(model, X_train, label_encoder, top_n=5):
    class_labels = label_encoder.classes_
    num_classes = len(class_labels)

    # Use SHAP's unified interface for better compatibility
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))
    axes = axes.flatten()
    for i, class_name in enumerate(class_labels):
        # print(f"Class: {class_name}")

        # SHAP values for this class
        class_shap_values = shap_values[i] if isinstance(shap_values, list) else shap_values[:, :, i] # Handle multi-output

        # Mean absolute SHAP for each feature
        mean_shap = np.abs(class_shap_values).mean(axis=0)

        # Top N features
        top_indices = np.argsort(mean_shap)[::-1][:top_n]
        top_features = X_train.columns[top_indices]
        top_values = mean_shap[top_indices]

        # Plot
        ax = axes[i]
        ax.barh(range(len(top_features)), top_values[::-1], align='center')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1])
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title(f"Top {top_n} Features for Class: {class_name}")
        ax.invert_yaxis()  # To display the most important feature at the top
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    return fig

def get_metrics(models,variant):
    model_variants_records = []
    for model_name, model in models.items():
        for variant_name, (Xr,yr,Xt,yt) in variant.items():
            model_variants_records.append(evaluate_model(model, Xr, yr, Xt, yt, f"{model_name}_{variant_name}"))
    
    return pd.DataFrame(model_variants_records)