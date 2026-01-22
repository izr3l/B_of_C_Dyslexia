import streamlit as st
import pandas as pd
import numpy as np
import os
from src import data_preprocessing, visualization, model_training

# Check if data exists
DATA_PATH = os.path.join(os.path.dirname(__file__), 'dyslexia_data.csv')

def main():
    st.set_page_config(page_title="Dyslexia Detection AI", layout="wide", page_icon="🧠")
    
    st.title("🧠 Dyslexia Detection using Machine Learning")
    st.markdown("### Early Detection Using Behavioral Data")

    # Load Data
    df = data_preprocessing.load_data(DATA_PATH)
    if df is None:
        st.error(f"Dataset not found at {DATA_PATH}. Please ensure 'dyslexia_data.csv' is in the project root.")
        return

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Project Info", "Exploratory Data Analysis", "Model Playground", "Model Comparison", "Live Prediction"])

    # Clean and Preprocess
    df_clean = data_preprocessing.clean_data(df)
    
    # Store processed data in session state to avoid re-running widely
    if 'preprocessed_data' not in st.session_state:
        X_train, X_test, y_train, y_test, scaler, feature_names = data_preprocessing.preprocess_data(df_clean)
        st.session_state['preprocessed_data'] = (X_train, X_test, y_train, y_test, scaler, feature_names)
    
    X_train, X_test, y_train, y_test, scaler, feature_names = st.session_state['preprocessed_data']
    trainer = model_training.ModelTrainer()

    # --- TAB 1: Project Info ---
    if options == "Project Info":
        st.header("Project Overview")
        with st.expander("📝 Project Description", expanded=True):
            st.markdown("""
            **Goal**: Detect the likelihood of dyslexia using behavioral metrics (reading speed, accuracy, reaction time).
            
            **Why?**: Dyslexia is often undiagnosed. AI can provide a low-cost, accessible screening tool.
            
            **Approach**: Supervised Learning (Classification) + Unsupervised Learning (Cluster Analysis).
            """)
        
        st.subheader("Team Members")
        st.markdown("""
        - Farhan Ahmad Nasiruddeen
        - Israel Oluwabukunmi Olayemi
        - Mahmud Yusuf Aminu
        - Makinde Mark Olusanya
        - Abdulmuhaimin Muhammad
        """)

        st.subheader("Dataset Snapshot")
        st.dataframe(df.head())

    # --- TAB 2: EDA ---
    elif options == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis (EDA)")
        
        col1, col2 = st.columns(2)
        with col1:
             st.plotly_chart(visualization.plot_target_distribution(df_clean, 'Dyslexia'), use_container_width=True)
        with col2:
             st.plotly_chart(visualization.plot_correlation_heatmap(df_clean.select_dtypes(include=[np.number])), use_container_width=True)

        st.subheader("Feature Distributions")
        selected_feature = st.selectbox("Select Feature to Visualize", df_clean.select_dtypes(include=[np.number]).columns)
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(visualization.plot_distribution(df_clean, selected_feature, mood='hist'), use_container_width=True)
        with c2:
            st.plotly_chart(visualization.plot_distribution(df_clean, selected_feature, mood='box'), use_container_width=True)

    # --- TAB 3: Model Playground ---
    elif options == "Model Playground":
        st.header("🛠 Train & Test Models")
        st.markdown("Adjust parameters and evaluate performance in real-time.")

        model_choice = st.selectbox("Select Model", ["KNN", "SVM", "Decision Tree", "Neural Network (MLP)", "Linear Regression", "K-Means"])
        
        metrics = None
        model = None
        
        if model_choice == "Decision Tree":
            depth = st.slider("Max Depth", 1, 20, 5)
            criterion = st.selectbox("Criterion", ["gini", "entropy"])
            if st.button("Train Decision Tree"):
                model = trainer.train_decision_tree(X_train, y_train, max_depth=depth, criterion=criterion)
                metrics = trainer.evaluate_model(model, X_test, y_test, model_type='classifier')

        elif model_choice == "SVM":
            C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            if st.button("Train SVM"):
                model = trainer.train_svm(X_train, y_train, C=C, kernel=kernel)
                metrics = trainer.evaluate_model(model, X_test, y_test, model_type='classifier')

        elif model_choice == "KNN":
            k = st.slider("K (Neighbors)", 1, 20, 5)
            if st.button("Train KNN"):
                model = trainer.train_knn(X_train, y_train, k=k)
                metrics = trainer.evaluate_model(model, X_test, y_test, model_type='classifier')

        elif model_choice == "Neural Network (MLP)":
            hidden_layers = st.selectbox("Hidden Layer Config", [(50,), (100,), (100, 50), (50, 50, 50)])
            max_iter = st.number_input("Max Iterations", 200, 1000, 500)
            if st.button("Train ANN"):
                model = trainer.train_ann(X_train, y_train, hidden_layer_sizes=hidden_layers, max_iter=max_iter)
                metrics = trainer.evaluate_model(model, X_test, y_test, model_type='classifier')

        elif model_choice == "Linear Regression":
            st.info("Linear Regression output will be thresholded at 0.5 for classification.")
            if st.button("Train Linear Regression"):
                model = trainer.train_linear_regression(X_train, y_train)
                metrics = trainer.evaluate_model(model, X_test, y_test, model_type='regression')

        elif model_choice == "K-Means":
            n_clusters = st.slider("Number of Clusters", 2, 5, 2)
            if st.button("Run K-Means"):
                model = trainer.train_kmeans(X_train, n_clusters=n_clusters)
                st.success("K-Means clustering complete.")
                
                # Visualize Cluster Separation (using first 2 valid features or PCA if we had it)
                # For simplicity, we create a scatter plot of first two features colored by cluster label
                labels = model.predict(X_test)
                
                # Create a temporary df for plotting
                plot_df = X_test.iloc[:, :2].copy()
                plot_df['Cluster'] = labels.astype(str)
                fig = visualization.plot_scatter(plot_df, plot_df.columns[0], plot_df.columns[1], 'Cluster')
                st.plotly_chart(fig)
                
                st.info("Note: K-Means is unsupervised. Colors represent discovered clusters, not necessarily Dyslexia labels.")

        if metrics:
            st.success("Training Complete!")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['Precision']:.4f}")
            col3.metric("Recall", f"{metrics['Recall']:.4f}")
            col4.metric("F1 Score", f"{metrics['F1']:.4f}")
            
            st.subheader("Confusion Matrix")
            st.plotly_chart(visualization.plot_confusion_matrix(metrics['Confusion Matrix'], labels=["No", "Yes"]), use_container_width=True)

    # --- TAB 4: Comparison ---
    elif options == "Model Comparison":
        st.header("🏆 Model Leaderboard")
        if st.button("Train & Compare All Models"):
            with st.spinner("Training all models..."):
                results = []
                
                # KNN
                m_knn = trainer.train_knn(X_train, y_train)
                res_knn = trainer.evaluate_model(m_knn, X_test, y_test)
                res_knn['Model'] = "KNN"
                results.append(res_knn)
                
                # SVM
                m_svm = trainer.train_svm(X_train, y_train)
                res_svm = trainer.evaluate_model(m_svm, X_test, y_test)
                res_svm['Model'] = "SVM"
                results.append(res_svm)
                
                # DT
                m_dt = trainer.train_decision_tree(X_train, y_train)
                res_dt = trainer.evaluate_model(m_dt, X_test, y_test)
                res_dt['Model'] = "Decision Tree"
                results.append(res_dt)
                
                # ANN
                m_ann = trainer.train_ann(X_train, y_train)
                res_ann = trainer.evaluate_model(m_ann, X_test, y_test)
                res_ann['Model'] = "ANN (MLP)"
                results.append(res_ann)

                # Linear Regression
                m_lr = trainer.train_linear_regression(X_train, y_train)
                res_lr = trainer.evaluate_model(m_lr, X_test, y_test, model_type='regression')
                res_lr['Model'] = "Linear Reg"
                results.append(res_lr)
                
                results_df = pd.DataFrame(results)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1']]
                st.session_state['comparison_results'] = results_df
        
        if 'comparison_results' in st.session_state:
            st.dataframe(st.session_state['comparison_results'].style.highlight_max(axis=0))
            st.plotly_chart(visualization.plot_model_comparison(st.session_state['comparison_results']), use_container_width=True)

    # --- TAB 5: Live Prediction ---
    elif options == "Live Prediction":
        st.header("🔮 Make a Prediction")
        st.markdown("Enter values to predict if a user is at risk.")
        
        # We need to collect inputs corresponding to X_train columns
        input_data = {}
        cols = st.columns(3)
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                # Assuming standard scaler, we input raw values and scale them later
                # Using slider for more intuitive input
                val = st.slider(f"{feature}", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
                input_data[feature] = val
        
        if st.button("Predict"):
            # Create DF
            input_df = pd.DataFrame([input_data])
            # Scale
            input_scaled = scaler.transform(input_df)
            
            # Use a default robust model for prediction (e.g., SVM or Decision Tree)
            # In a real scenario, we'd select the "Best" saved model.
            # Here we just re-train a quick DT for demonstration or use one from session if saved.
            # Let's train a quick Decision Tree on full data for best simple performance
            model = trainer.train_decision_tree(X_train, y_train) 
            
            prediction = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]
            
            st.markdown("---")
            if prediction == 1:
                st.error(f"**Prediction: High Risk of Dyslexia** (Probability: {prob:.2f})")
            else:
                st.success(f"**Prediction: Low Risk** (Probability: {prob:.2f})")

if __name__ == "__main__":
    main()
