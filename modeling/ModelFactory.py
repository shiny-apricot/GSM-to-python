# model_factory.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

class ModelFactory:
    def __init__(self):
        """
        Initialize the ModelFactory class.
        """
        self.models = {
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "logistic_regression": LogisticRegression,
            "svm": SVC,
            "knn": KNeighborsClassifier,
            "decision_tree": DecisionTreeClassifier
        }

    def create_model(self, model_name, **kwargs):
        """
        Create a machine learning model based on the given model name.

        Args:
            model_name (str): The name of the model to create.
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Returns:
            object: The created machine learning model.
        """
        if model_name not in self.models:
            raise ValueError(f"Unsupported model name: {model_name}")
        return self.models[model_name](**kwargs)

    def get_supported_models(self):
        """
        Get a list of supported model names.

        Returns:
            list: A list of supported model names.
        """
        return list(self.models.keys())