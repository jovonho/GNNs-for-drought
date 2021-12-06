"""
Training a benchmark model
(a multi output random forest regressor)
against the dataset
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.data import Dataset
from src.utils import filter_preds_test_by_mask


def main():
    print("Training a multi-output random forest")

    train_dataset = Dataset(is_test=False)
    test_dataset = Dataset(is_test=True)
    print(train_dataset.cached_dynamic_means_and_stds)

    # values taken from
    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_random_forest_regression_multioutput.html#sphx-glr-auto-examples-ensemble-plot-random-forest-regression-multioutput-py
    max_depth = 3
    n_jobs = 4  # -1 to use all available cores
    rf = RandomForestRegressor(
        n_estimators=10,
        max_depth=max_depth,
        random_state=0,
        verbose=1,
        n_jobs=n_jobs,
    )

    train_x, train_y_org, mask = train_dataset.load_all_data()
    print(f"\nLoaded training data! X: {train_x.shape}, Y: {train_y_org.shape}. Fitting the model")

    train_x, train_y = filter_preds_test_by_mask(train_x, train_y_org, mask)

    rf.fit(train_x, train_y)

    print("Finished fitting! Evaluating on the test years")
    test_x, test_y, test_mask = test_dataset.load_all_data()
    test_preds = rf.predict(test_x)
    test_preds, test_y = filter_preds_test_by_mask(test_preds, test_y, test_mask)

    print(test_preds.shape)

    print(f"MSE: {mean_squared_error(test_y, test_preds)}")
    print(f"R2: {r2_score(test_y, test_preds)}")


if __name__ == "__main__":
    main()
