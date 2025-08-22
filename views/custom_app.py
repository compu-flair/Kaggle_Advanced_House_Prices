import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from configs.config import CUSTOM_APP_TAB

def custom_model_linear_regression_app():
    st.title("Custom Model Linear Regression Training")
    st.session_state["selected_tab"] = CUSTOM_APP_TAB

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())


        # Display information about nun values, median and means of each column
        st.subheader("Data Summary")
        st.write("Number of missing values:")
        st.write(df.isnull().sum())
        st.write("Median values:")
        st.write(df.select_dtypes(include=['number']).median())
        st.write("Mean values:")
        st.write(df.select_dtypes(include=['number']).mean())

        # Calculate and display correlation matrix
        st.subheader("Correlation Matrix")
        corr_matrix = df.corr(numeric_only=True)
        st.write(corr_matrix)
        fig = plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8})
        st.pyplot(fig)

        target = st.selectbox("Select label (Y) column", df.columns)
        features = st.multiselect(
            "Select feature columns (X)", [col for col in df.columns if col != target]
        )

        if features and target:
            # Store model and feature info in session state to persist after training
            if "trained_model" not in st.session_state:
                st.session_state["trained_model"] = None
                st.session_state["numeric_features"] = []
                st.session_state["categorical_features"] = []

            if st.button("Start Training", key="start_training_btn"):
                # Clean data: drop rows with NaN values in selected features or target
                clean_df = df[features + [target]].dropna()
                X = clean_df[features]
                y = clean_df[target]

                # Identify column types
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

                # Preprocessing
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                    ]
                )

                # Pipeline
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', LinearRegression())
                ])

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)

                # Predictions
                y_pred = model.predict(X_test)

                # Evaluation metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                st.write(f"Model R^2 score on test set: {score:.4f}")
                st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                st.write(f"Mean Squared Error (MSE): {mse:.4f}")

                # Save model and feature info in session state
                st.session_state["trained_model"] = model
                st.session_state["numeric_features"] = numeric_features
                st.session_state["categorical_features"] = categorical_features
                st.session_state["selected_features"] = features
                st.session_state["df"] = df

            # Show prediction section if model is trained
            if st.session_state.get("trained_model") is not None:
                st.subheader("Make a Prediction")
                input_data = {}
                df_for_input = st.session_state["df"]
                numeric_features = st.session_state["numeric_features"]
                categorical_features = st.session_state["categorical_features"]
                selected_features = st.session_state["selected_features"]

                for feature in selected_features:
                    if feature in numeric_features:
                        val = st.number_input(f"Input value for {feature}", value=float(df_for_input[feature].mean()))
                    else:
                        val = st.selectbox(f"Input value for {feature}", options=df_for_input[feature].unique())
                    input_data[feature] = val

                if "prediction_result" not in st.session_state:
                    st.session_state["prediction_result"] = None

                if st.button("Predict", key="predict_btn"):
                    input_df = pd.DataFrame([input_data])
                    pred = st.session_state["trained_model"].predict(input_df)[0]
                    st.session_state["prediction_result"] = f"Predicted value: {pred:.4f}"

                if st.session_state.get("prediction_result"):
                    st.write(st.session_state["prediction_result"])