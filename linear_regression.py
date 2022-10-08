import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics


class LinearRegressionMethod():
    def __init__(self):
        self.products = pd.read_csv("Cleaned_Products.csv", lineterminator="\n")
        
    
    def run_data(self):
        X = self.products.loc[:, ["product_name", "product_description", "location"]]
        
        y = self.products['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        transformer = ColumnTransformer(
            [('vect1', TfidfVectorizer(), 'product_name'),
            ('vect2', TfidfVectorizer(), 'product_description'),
            ('vect3', TfidfVectorizer(), 'location')],
            remainder='passthrough'
        )
        
        # define a pipeline
        pipe = Pipeline(
        [
            ("vect", transformer),
            ("tfidf", TfidfTransformer()),
            ("reg", LinearRegression()),
        ]
        )
        print(f'Making Pipeline : {pipe}')

        parameters = {
            'vect__vect1__ngram_range': ((1, 1), (1, 2)),
            'vect__vect1__min_df': (0.005, 0.008, 0.01),
            'vect__vect2__ngram_range': ((1, 1), (1, 2), (1, 3)),
            'vect__vect2__min_df': (0.005, 0.008, 0.01),
            # 'vect__max_features': (None, 5000, 10000, 50000),
            'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
            "reg__fit_intercept": (True, False),
        }

        grid_search = GridSearchCV(pipe, parameters, n_jobs=-1, verbose=1)

       # Fit gridsearch
        fit = grid_search.fit(X_train, y_train)
        print(f"Fitting pipeline: {fit}")

        # Make prediction based on pipeline:
        y_pred = grid_search.predict(X_test)
        print(f'Making predictions: {y_pred}')

        # Print Predictions
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', metrics.r2_score(y_test, y_pred))

    def run_linear_regress(self):
        self.run_data()
        


if __name__ == "__main__":
    LinearRegress = LinearRegressionMethod()
    LinearRegress.run_linear_regress()