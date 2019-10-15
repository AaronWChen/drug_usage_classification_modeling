import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor



def bag_model(df, target, depth, samples):

    X = df.drop([target], axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   random_state=1)

    rt = DecisionTreeRegressor(random_state=1, max_depth=depth)

    rf = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=1)

    bag = BaggingRegressor(n_estimators=100,
                       max_features=X.shape[1],
                       max_samples=samples,
                       random_state=1)

    rt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    bag.fit(X_train, y_train)

    return f"""Decision Tree Score = {rt.score(X_test, y_test)}, ---
    Bagging Score = {bag.score(X_test, y_test)}, --- Random Forest Score
    {rf.score(X_test, y_test)}"""