"""
Training a benchmark model
(a multi output random forest regressor)
against the dataset
"""
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.data import Dataset


def main():
    print("Training a multi-output random forest")

    train_dataset = Dataset(is_test=False)
    test_dataset = Dataset(is_test=True)

    # values taken from
    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_random_forest_regression_multioutput.html#sphx-glr-auto-examples-ensemble-plot-random-forest-regression-multioutput-py
    max_depth = 3
    n_jobs = 4  # -1 to use all available cores
    regr_multirf = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=10,
            max_depth=max_depth,
            random_state=0,
            verbose=1,
            n_jobs=n_jobs,
        )
    )

    train_x, train_y, _ = train_dataset.load_all_data()
    print(f"\nLoaded training data! X: {train_x.shape}, Y: {train_y.shape}. Fitting the model")

    regr_multirf.fit(train_x, train_y)

    print("Finished fitting! Evaluating on the test years")
    test_x, test_y, _ = test_dataset.load_all_data()
    test_preds = regr_multirf.predict(test_x)

    print(test_preds.shape)

    print(f"MSE: {mean_squared_error(test_y, test_preds)}")


if __name__ == "__main__":
    main()
