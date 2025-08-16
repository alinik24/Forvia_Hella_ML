import h2o
import pandas as pd
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.frame import H2OFrame


def train_gbm_model(X_train, y_train, X_test, y_test, seed=42):
    # Convert Pandas to H2O
    train_h2o = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
    test_h2o = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

    target = y_train.name
    features = [col for col in train_h2o.columns if col != target]

    # Ensure target is categorical
    train_h2o[target] = train_h2o[target].asfactor()
    test_h2o[target] = test_h2o[target].asfactor()

    # Define model
    model = H2OGradientBoostingEstimator(
        distribution="multinomial",
        balance_classes=True,
        ntrees=100,
        max_depth=6,
        learn_rate=0.1,
        seed=seed
    )

    # Train
    model.train(x=features, y=target, training_frame=train_h2o)

    # Evaluate
    performance = model.model_performance(test_data=test_h2o)
    print(performance)

    return model, features


def plot_feature_importance(model, max_features=20):
    model.varimp_plot(num_of_features=max_features)
