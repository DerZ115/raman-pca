import numpy as np
import pandas as pd
import shap
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (GridSearchCV,
                                     ParameterGrid,
                                     StratifiedKFold,
                                     cross_validate,
                                     cross_val_predict)
from mlxtend.evaluate import mcnemar_table, mcnemar
from .misc import mode
from tqdm.notebook import tqdm



def randomized_cv(clf, X, y, param_grid,
                  gs_scoring="accuracy",
                  ct_scoring="accuracy",
                  n_folds=10,
                  n_trials=20,
                  wns=None,
                  y_names=None,
                  explainer=None,
                  n_jobs=-1,
                  verbose=0):

    combs = list(ParameterGrid(param_grid))

    ct_train_scores = {score: np.zeros(
        (n_trials, n_folds)) for score in ct_scoring}
    ct_test_scores = {score: np.zeros((n_trials, n_folds))
                      for score in ct_scoring}

    gs_train_scores = np.zeros([n_trials, len(combs)])
    gs_test_scores = np.zeros([n_trials, len(combs)])

    opt_params = {param: np.zeros_like(
        vals, shape=n_trials) for param, vals in param_grid.items()}

    conf_scores = np.zeros((n_trials, len(y)))
    y_pred = np.zeros((n_trials, len(y)))

    p_vals = np.zeros((n_trials))

    shap_vals = np.zeros(n_trials, dtype=object)

    for i in tqdm(range(n_trials)):  # Random permutations

        outer_cv = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=i)
        inner_cv = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=i)

        # hyperparameter optimization
        gridsearch = GridSearchCV(clf,
                                  param_grid=param_grid,
                                  cv=inner_cv,
                                  scoring=gs_scoring,
                                  return_train_score=True,
                                  verbose=verbose,
                                  n_jobs=n_jobs)

        gridsearch.fit(X, y)

        for param, value in gridsearch.best_params_.items():
            opt_params[param][i] = value

        gs_train_scores[i] = gridsearch.cv_results_["mean_train_score"]
        gs_test_scores[i] = gridsearch.cv_results_["mean_test_score"]

        gridsearch.set_params(**{"n_jobs": 1})

        ct_results = cross_validate(gridsearch,
                                    X, y,
                                    scoring=ct_scoring,
                                    cv=outer_cv,
                                    return_estimator=True,
                                    return_train_score=True,
                                    n_jobs=n_jobs,
                                    verbose=verbose)

        dummy_results = cross_val_predict(DummyClassifier(), X, y, cv=outer_cv)

        for score in ct_scoring:
            ct_test_scores[score][i, :] = ct_results["test_" + score]
            ct_train_scores[score][i, :] = ct_results["train_" + score]

        shap_vals_current = np.zeros((len(y), X.shape[1]))
        shap_base_vals_current = np.zeros(len(y))
        shap_data_current = np.zeros((len(y), X.shape[1]))

        for j, (train, test) in enumerate(outer_cv.split(X, y)):
            X_test = X[test]
            X_train = X[train]

            gs = ct_results["estimator"][j]

            if explainer is not None:
                if isinstance(gs.best_estimator_, Pipeline):
                    current_clf = gs.best_estimator_[-1]
                    current_prep = gs.best_estimator_[:-1]

                X_train_prep = current_prep.transform(X_train)
                X_test_prep = current_prep.transform(X_test)
                explainer = shap.Explainer(current_clf, X_train_prep)
                shap_tmp = explainer(X_test_prep)
                shap_vals_current[test, :] = shap_tmp.values
                shap_base_vals_current[test] = shap_tmp.base_values
                shap_data_current[test, :] = shap_tmp.data

            y_pred[i, test] = gs.predict(X_test)

            try:
                tmp = gs.predict_proba(X_test)[:, 1]
            except AttributeError:
                tmp = gs.decision_function(X_test)

            conf_scores[i, test] = tmp

        mcn_table = mcnemar_table(
            y.ravel(), dummy_results.ravel(), y_pred[i, :].ravel())
        _, p_vals[i] = mcnemar(mcn_table)

        shap_vals[i] = shap.Explanation(shap_vals_current,
                                        base_values=shap_base_vals_current,
                                        data=shap_data_current,
                                        feature_names=wns,
                                        output_names=y_names)

    ct_results = {
        "train_scores": ct_train_scores,
        "test_scores": ct_train_scores,
        "y_pred": y_pred,
        "conf_scores": conf_scores,
        "p_vals": p_vals,
        "shap_vals": shap_vals
    }

    cv_results = {
        "combinations": combs,
        "train_scores": gs_train_scores,
        "test_scores": gs_test_scores,
        "opt_params": opt_params
    }

    print(f"Training Accuracy: {ct_train_scores['accuracy'].mean():.3f}")
    print(f"Training F1-score: {ct_train_scores['f1'].mean():.3f}")
    print()
    print(f"Test Accuracy: {ct_test_scores['accuracy'].mean():.3f}")
    print(f"Test F1-score: {ct_test_scores['f1'].mean():.3f}")

    params_final_model = {}

    for param, vals in opt_params.items():
        if vals.dtype == float:
            val_final = np.mean(vals)
        elif vals.dtype == int:
            val_final = np.median(vals)
            if int(val_final) == val_final:
                val_final = int(val_final)
            else:
                val_final = mode(vals)
        else:
            val_final = mode(vals)

        params_final_model[param] = val_final

    clf_final = clone(clf)

    clf_final.set_params(**params_final_model)
    clf_final.fit(X, y)

    return clf_final, ct_results, cv_results
