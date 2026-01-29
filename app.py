import streamlit as st
import pandas as pd
import numpy as np
import os
from src import data_preprocessing, visualization, model_training

# Check if data exists
DATA_PATH = os.path.join(os.path.dirname(__file__), 'dyslexia_synthetic_4237.csv')

def main():
    st.set_page_config(page_title="Dyslexia Detection AI", layout="wide", page_icon="🧠")
    
    st.title("Dyslexia Detection using Machine Learning")
    st.markdown("### Early Detection Using Behavioral Data")

    # Load Data
    df = data_preprocessing.load_data(DATA_PATH)
    if df is None:
        st.error(f"Dataset not found at {DATA_PATH}. Please ensure 'dyslexia_synthetic_4237.csv' is in the project root.")
        return

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Project Info", "Exploratory Data Analysis", "Model Playground", "Model Comparison", "Live Prediction"])

    # Clean and Preprocess
    df_clean = data_preprocessing.clean_data(df)
    
    # Store processed data in session state to avoid re-running widely
    # Store processed data in session state to avoid re-running widely
    # Updated key to v3 to force reload with new dataset
    if 'preprocessed_data_v3' not in st.session_state:
        X_train, X_test, y_train, y_test, scaler, feature_names = data_preprocessing.preprocess_data(df_clean)
        st.session_state['preprocessed_data_v3'] = (X_train, X_test, y_train, y_test, scaler, feature_names)
    
    X_train, X_test, y_train, y_test, scaler, feature_names = st.session_state['preprocessed_data_v3']
    trainer = model_training.ModelTrainer()

    # --- TAB 1: Project Info ---
    if options == "Project Info":
        st.header("Project Overview")
        with st.expander("Project Description", expanded=True):
            st.markdown("""
            **Goal**: Detect the likelihood of dyslexia using behavioral metrics (Complex Processing, Reading Recognition, Memory Task, Basic Attention Task).
            
            **Why?**: Dyslexia is often undiagnosed. AI can provide a low-cost, accessible screening tool.
            
            **Approach**: Supervised Learning (Classification)
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
        st.subheader("Train & Test Models")
        st.markdown("Adjust parameters and evaluate performance in real-time.")

        model_choice = st.selectbox("Select Model", ["Random Forest", "SVM"])
        
        metrics = None
        
        if model_choice == "Random Forest":
            n_estimators = st.slider("Numbers of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 1, 20, 10)
            if st.button("Train Random Forest"):
                model = trainer.train_random_forest(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth)
                metrics = trainer.evaluate_model(model, X_test, y_test, model_type='classifier')

        elif model_choice == "SVM":
            C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            if st.button("Train SVM"):
                model = trainer.train_svm(X_train, y_train, C=C, kernel=kernel)
                metrics = trainer.evaluate_model(model, X_test, y_test, model_type='classifier')

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
        st.header("Model Leaderboard")
        if st.button("Train & Compare All Models"):
            with st.spinner("Training all models..."):
                results = []
                
                # Random Forest
                m_rf = trainer.train_random_forest(X_train, y_train)
                res_rf = trainer.evaluate_model(m_rf, X_test, y_test)
                res_rf['Model'] = "Random Forest"
                results.append(res_rf)
                
                # SVM
                m_svm = trainer.train_svm(X_train, y_train)
                res_svm = trainer.evaluate_model(m_svm, X_test, y_test)
                res_svm['Model'] = "SVM"
                results.append(res_svm)
                
                results_df = pd.DataFrame(results)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1']]
                st.session_state['comparison_results'] = results_df
        
        if 'comparison_results' in st.session_state:
            st.dataframe(st.session_state['comparison_results'].style.highlight_max(axis=0))
            st.plotly_chart(visualization.plot_model_comparison(st.session_state['comparison_results']), use_container_width=True)

    # --- TAB 5: Live Prediction ---
    elif options == "Live Prediction":
        st.header("Make a Prediction")
        st.markdown("Enter behavioral test values to predict dyslexia risk.")
        
        # Get the original (unscaled) data to determine proper min/max ranges
        df_for_ranges = df_clean.drop(columns=['Dyslexia'])
        
        # Collect inputs with proper ranges based on actual data
        input_data = {}
        
        # Organize features by category for better UX
        st.subheader("Demographics")
        demo_cols = st.columns(2)
        with demo_cols[0]:
            if 'Gender' in feature_names:
                gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                input_data['Gender'] = gender
        with demo_cols[1]:
            if 'Age' in feature_names:
                age_min = int(df_for_ranges['Age'].min())
                age_max = int(df_for_ranges['Age'].max())
                age = st.slider("Age", min_value=age_min, max_value=age_max, value=(age_min + age_max) // 2)
                input_data['Age'] = age
        
        # Test metrics - organized by test number
        test_numbers = ['4', '12', '26', '27']
        test_descriptions = {
            '4': "Test 4 - Basic Attention Task",
            '12': "Test 12 - Memory Task", 
            '26': "Test 26 - Reading Recognition",
            '27': "Test 27 - Complex Processing"
        }
        
        for test_num in test_numbers:
            # Check if this test exists in our dataset
            if any(f.endswith(test_num) for f in feature_names):
                st.subheader(f"{test_descriptions.get(test_num, f'Test {test_num}')}")
                cols = st.columns(3)
                
                with cols[0]:
                    # Using averages from the dataset as defaults (e.g., ~4)
                    hits = st.slider(f"Hits ({test_num})", 0, 100, 4, key=f"h_{test_num}")
                
                with cols[1]:
                    # Using averages from the dataset as defaults (e.g., ~2)
                    misses = st.slider(f"Misses ({test_num})", 0, 100, 2, key=f"m_{test_num}")
                
                # Clicks calculated automatically (User requested: Total click = Hits + Misses)
                clicks = hits + misses
                
                with cols[2]:
                    # Display the calculated total clicks
                    st.metric(f"Total Clicks ({test_num})", clicks)
                
                # Automatic Calculations
                score = hits # In this dataset, Score is identity with Hits
                accuracy = hits / clicks if clicks > 0 else 0.0
                missrate = misses / clicks if clicks > 0 else 0.0
                
                # Store all required features for the model
                input_data[f"Hits{test_num}"] = hits
                input_data[f"Misses{test_num}"] = misses
                input_data[f"Clicks{test_num}"] = clicks
                input_data[f"Score{test_num}"] = score
                input_data[f"Accuracy{test_num}"] = accuracy
                input_data[f"Missrate{test_num}"] = missrate
                
                # Optional: Show the calculated values to the user
                st.caption(f"Calculated: Accuracy={accuracy:.2f}, Missrate={missrate:.2f}")
        
        st.markdown("---")
        
        # Model selection
        st.subheader("Select Prediction Model")
        model_options = {
            "Random Forest": "Balanced & robust ensemble (Recommended)",
            "SVM": "Good for complex decision boundaries"
        }
        
        selected_model = st.selectbox(
            "Choose Model",
            options=list(model_options.keys()),
            help="Different models may give different predictions"
        )
        st.caption(f"ℹ️ {model_options[selected_model]}")
        
        # Train and cache models (only train if not already cached)
        model_cache_key = f'prediction_model_{selected_model}'
        
        if model_cache_key not in st.session_state:
            with st.spinner(f"Training {selected_model}..."):
                if selected_model == "Random Forest":
                    st.session_state[model_cache_key] = trainer.train_random_forest(X_train, y_train)
                elif selected_model == "SVM":
                    # Using higher C to penalize misclassifications on minority class
                    st.session_state[model_cache_key] = trainer.train_svm(X_train, y_train, C=10.0, kernel='rbf')
        
        # Debug / Info Expander
        with st.expander("Prediction Technical Details"):
            st.write(f"**Model Type**: {selected_model}")
            st.write(f"**Total Features**: {len(feature_names)}")
            st.json(input_data)
        
        if st.button("Predict Risk", type="primary"):
            # Create DataFrame with features in correct order
            input_df = pd.DataFrame([input_data])[list(feature_names)]
            
            # Scale using the fitted scaler
            input_scaled = scaler.transform(input_df)
            
            # Use the selected cached model
            model = st.session_state[model_cache_key]
            
            prediction = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]
            
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Visual risk indicator
            # The model is trained to spot Dyslexia=1. 
            # A higher probability means higher likelihood of Dyslexia.
            risk_percentage = prob * 100
            
            # Using broader thresholds for 'Moderate' to catch edge cases
            if risk_percentage < 30:
                risk_level = "Low"
                risk_color = "🟢"
            elif risk_percentage < 70:
                risk_level = "Moderate"
                risk_color = "🟡"
            else:
                risk_level = "High"
                risk_color = "🔴"
            
            # Display risk gauge
            st.markdown(f"### {risk_color} Risk Level: **{risk_level}**")
            st.progress(prob, text=f"Dyslexia Probability: {risk_percentage:.1f}%")
            
            # Detailed breakdown
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Probability", f"{risk_percentage:.1f}%")
            with col2:
                st.metric("Risk Category", risk_level)
            with col3:
                # Prediction is 1 if prob > 0.5 usually, but we can be more sensitive
                pred_label = "At Risk" if prob > 0.5 else "Not At Risk"
                st.metric("Prediction", pred_label)
            
            st.markdown("---")
            
            # Interpretation
            if prediction == 1:
                st.error("⚠️ **Result: Indicators suggest potential dyslexia risk**")
                st.markdown("""
                **Recommended Next Steps:**
                - Consult with an educational psychologist
                - Consider formal dyslexia assessment
                - Discuss findings with teachers/educators
                """)
            else:
                st.success("✅ **Result: No significant dyslexia indicators detected**")
                st.markdown("""
                **Note:** This screening did not detect strong dyslexia markers based on the provided metrics.
                If concerns persist, professional evaluation is still recommended.
                """)
            
            st.caption("⚕️ *Disclaimer: This AI tool is for preliminary screening only and should not replace professional medical or educational diagnosis.*")

if __name__ == "__main__":
    main()
