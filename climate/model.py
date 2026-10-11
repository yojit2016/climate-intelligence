"""Three direct Random Forest models with honest held-out evaluation."""
from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from climate.evaluation import ModelUnavailable, chronological_split, persistence, scores
from climate.features import FEATURES, daily_series, make_features


@dataclass
class ForecastResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    splits: pd.DataFrame
    models: dict[int, RandomForestRegressor]


def forest(depth: int | None) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=160, max_depth=depth,
                                 min_samples_leaf=3, random_state=42, n_jobs=-1)


def train_and_predict(frame: pd.DataFrame) -> ForecastResult:
    series = daily_series(frame)
    forecasts, metrics, tests, splits, models = [], [], [], [], {}
    for horizon in (1, 2, 3):
        features = make_features(series, horizon)
        inference = features.iloc[[-1]]
        if inference[FEATURES].isna().any(axis=None):
            raise ModelUnavailable("The most recent 14 days must contain valid temperatures for forecasting.")
        samples = features.dropna(subset=FEATURES + ["target"])
        train, validation, test = chronological_split(samples)
        candidates = []
        for depth in (6, None):
            candidate = forest(depth)
            candidate.fit(train[FEATURES], train["target"])
            error = scores(validation["target"], candidate.predict(validation[FEATURES]))["mae"]
            candidates.append((error, candidate))
        _, selected = min(candidates, key=lambda item: item[0])
        for algorithm, predicted in (("Random Forest", selected.predict(validation[FEATURES])),
                                     ("Persistence", persistence(validation))):
            metrics.append({"horizon": horizon, "partition": "Validation", "model": algorithm,
                            **scores(validation["target"], predicted), "samples": len(validation)})
        # Refit selected settings only on labels known before the first test origin.
        before_test = samples.loc[(samples.index < test.index.min())
                                  & (samples["target_date"] < test.index.min())]
        selected.fit(before_test[FEATURES], before_test["target"])
        predictions = selected.predict(test[FEATURES])
        for algorithm, predicted in (("Random Forest", predictions), ("Persistence", persistence(test))):
            metrics.append({"horizon": horizon, "partition": "Test", "model": algorithm,
                            **scores(test["target"], predicted), "samples": len(test)})
        tests.append(pd.DataFrame({"date": test["target_date"], "actual": test["target"],
            "prediction": predictions, "baseline": persistence(test), "horizon": horizon}))
        for name, block in (("Train", train), ("Validation", validation), ("Test", test)):
            splits.append({"horizon": horizon, "partition": name, "rows": len(block),
                           "first_origin": block.index.min(), "last_origin": block.index.max(),
                           "last_target": block["target_date"].max()})
        # Production refit is separate from held-out scoring above.
        production = forest(selected.max_depth).fit(samples[FEATURES], samples["target"])
        models[horizon] = production
        forecasts.append({"date": inference["target_date"].iloc[0], "horizon": horizon,
                          "prediction": float(production.predict(inference[FEATURES])[0]),
                          "baseline": float(series.iloc[-1]), "origin": series.index[-1]})
    return ForecastResult(pd.DataFrame(forecasts), pd.DataFrame(metrics),
                          pd.concat(tests, ignore_index=True), pd.DataFrame(splits), models)
