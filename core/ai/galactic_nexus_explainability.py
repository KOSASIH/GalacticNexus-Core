import shap
 import lime

def generate_explanation_report(X_explain, y_explain, feature_names, class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_explain, feature_names=feature_names, class_names=class_names)
    explanation = explainer.explain_instance(X_explain, y_explain, num_features=5)
    return explanation
