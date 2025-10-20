import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Neural Network Classifier",
    page_icon="🧠",
    layout="wide"
)

# Model architecture
class ReuploadingNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        original_input = x
        x = self.layers[0](x)
        x = self.layers[1](x)
        for i in range(2, len(self.layers), 2):
            x = torch.cat([x, original_input], dim=1)
            x = self.layers[i](x)
            x = self.layers[i+1](x)
        return torch.sigmoid(self.output_layer(x))

# Load model
@st.cache_resource
def load_model():
    checkpoint = torch.load('best_model.pth', map_location='cpu')
    model = ReuploadingNN(
        checkpoint['input_dim'], 
        checkpoint['hyperparameters']['hidden_dim'], 
        checkpoint['hyperparameters']['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    norm_params = np.load('normalization_params.npz')
    return model, norm_params['mean'], norm_params['std'], checkpoint

model, mean, std, checkpoint = load_model()

# Title and description
st.title("🧠 Neural Network Signal Classifier")
st.markdown("### Classify signals as **Signal** or **Background**")

# Display model info
with st.expander("ℹ️ Model Information"):
    st.write(f"**Model Type:** ReuploadingNN")
    st.write(f"**Hidden Dimensions:** {checkpoint['hyperparameters']['hidden_dim']}")
    st.write(f"**Number of Layers:** {checkpoint['hyperparameters']['num_layers']}")
    st.write(f"**Validation AUC:** {checkpoint['val_auc']:.4f}")
    st.write(f"**Input Features:** {checkpoint['input_dim']}")

# Feature names
feature_names = [
    "10th Percentile", "90th Percentile", "Energy", "Entropy",
    "Interquartile Range", "Kurtosis", "Maximum", "Mean Absolute Deviation",
    "Mean", "Median", "Minimum", "Range", "Robust Mean Absolute Deviation",
    "Root Mean Squared", "Skewness", "Total Energy", "Uniformity", "Variance"
]

# Two tabs: Manual input and CSV upload
tab1, tab2 = st.tabs(["📝 Manual Input", "📁 Upload CSV"])

with tab1:
    st.markdown("#### Enter feature values:")

    # Create 3 columns for inputs
    cols = st.columns(3)
    features = []

    for i, name in enumerate(feature_names):
        with cols[i % 3]:
            value = st.number_input(
                name, 
                value=0.0, 
                format="%.6f",
                key=f"feature_{i}"
            )
            features.append(value)

    if st.button("🎯 Predict", type="primary", use_container_width=True):
        # Make prediction
        features_array = np.array(features, dtype=np.float32)
        features_normalized = (features_array - mean) / std

        with torch.no_grad():
            input_tensor = torch.tensor(features_normalized, dtype=torch.float32).unsqueeze(0)
            probability = model(input_tensor).item()

        prediction_class = 'Signal' if probability > 0.5 else 'Background'
        confidence = abs(probability - 0.5) * 2

        # Display results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction_class == 'Signal':
                st.success(f"### 🎯 {prediction_class}")
            else:
                st.info(f"### 🔵 {prediction_class}")

        with col2:
            st.metric("Probability", f"{probability:.4f}")

        with col3:
            st.metric("Confidence", f"{confidence:.2%}")

        st.progress(probability)

with tab2:
    st.markdown("#### Upload a CSV file")
    st.info("📌 If your CSV has labels in the first column, they will be automatically removed")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Remove non-numeric columns (like 'signal', 'Subject', etc.)
        df_numeric = df.select_dtypes(include=[np.number])

        # If first column looks like labels (0/1), skip it
        if df_numeric.shape[1] > 18:
            st.info("Detected label column - removing it")
            df_numeric = df_numeric.iloc[:, 1:]

        st.write(f"Loaded {len(df_numeric)} samples with {df_numeric.shape[1]} features")
        st.dataframe(df_numeric.head())

        if df_numeric.shape[1] != 18:
            st.error(f"⚠️ Expected 18 features, but got {df_numeric.shape[1]}. Please check your CSV.")
        else:
            if st.button("🎯 Predict All", type="primary"):
                predictions = []
                progress_bar = st.progress(0)

                for idx, row in df_numeric.iterrows():
                    features_array = row.values.astype(np.float32)
                    features_normalized = (features_array - mean) / std

                    with torch.no_grad():
                        input_tensor = torch.tensor(features_normalized, dtype=torch.float32).unsqueeze(0)
                        probability = model(input_tensor).item()

                    prediction_class = 'Signal' if probability > 0.5 else 'Background'
                    predictions.append({
                        'Sample': idx + 1,
                        'Class': prediction_class,
                        'Probability': probability
                    })

                    progress_bar.progress((idx + 1) / len(df_numeric))

                results_df = pd.DataFrame(predictions)
                st.markdown("### 📊 Batch Prediction Results")
                st.dataframe(results_df)

                signal_count = len(results_df[results_df['Class'] == 'Signal'])
                background_count = len(results_df[results_df['Class'] == 'Background'])

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🎯 Signal", signal_count)
                with col2:
                    st.metric("🔵 Background", background_count)

                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "predictions.csv",
                    "text/csv",
                    key='download-csv'
                )

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
