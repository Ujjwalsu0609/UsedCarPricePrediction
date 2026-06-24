import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(model , feature_names):
   importance = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
   importance = importance.sort_values(by="Importance",ascending=False)
   fig, ax = plt.subplots(figsize=(8,5))
   ax.barh(importance["Feature"],importance["Importance"])
   ax.set_title("Feature Importance")
   return fig