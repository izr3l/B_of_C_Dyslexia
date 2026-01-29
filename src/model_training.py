import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelTrainer:
    def __init__(self):
        pass

    def train_svm(self, X_train, y_train, C=1.0, kernel='rbf'):
        # Ensuring class_weight='balanced' and probability=True for reliable risk scores
        model = SVC(
            C=C, 
            kernel=kernel, 
            probability=True, 
            class_weight='balanced', 
            random_state=42,
            gamma='scale' # Default but explicit
        )
        model.fit(X_train, y_train)
        return model

    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=10):
        """Random Forest provides better probability estimates than single Decision Tree."""
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test, model_type='classifier', threshold=0.5):
        """
        Evaluates the model and returns metrics.
        model_type: 'classifier', 'regression', 'cluster'
        """
        metrics = {}
        
        if model_type == 'cluster':
            # For KMeans, we can't easily do standard accuracy without mapping clusters labels.
            # We will just return the cluster labels for visualization.
            y_pred = model.predict(X_test)
            metrics['y_pred'] = y_pred
            return metrics

        if model_type == 'regression':
            y_pred_cont = model.predict(X_test)
            y_pred = (y_pred_cont > threshold).astype(int)
        else:
            y_pred = model.predict(X_test)
            
        metrics['Accuracy'] = accuracy_score(y_test, y_pred)
        metrics['Precision'] = precision_score(y_test, y_pred, zero_division=0)
        metrics['Recall'] = recall_score(y_test, y_pred, zero_division=0)
        metrics['F1'] = f1_score(y_test, y_pred, zero_division=0)
        metrics['Confusion Matrix'] = confusion_matrix(y_test, y_pred)
        metrics['y_pred'] = y_pred
        
        return metrics
