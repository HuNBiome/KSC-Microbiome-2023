#!/usr/bin/env python3

import argparse
import datetime
import pandas as pd
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from joblib import dump, load

parser = argparse.ArgumentParser()
parser.add_argument('table_file')
parser.add_argument('y_feature')
parser.add_argument('model_file')
parser.add_argument('--test_size', dest = 'test_size', default = 0.2, type = float)
parser.add_argument('--eval_metric', dest = 'eval_metric', default = 'Accuracy')
parser.add_argument('--custom_metric', dest = 'custom_metric', action = 'append')
parser.add_argument('--thread_count', dest = 'thread_count', default = 1, type = int)
parser.add_argument('--used_ram_limit', dest = 'used_ram_limit')
parser.add_argument('--cat_feature', dest = 'cat_feature', action = 'append')
parser.add_argument('--study_direction', dest = 'study_direction', default = 'maximize', choices = ['maximize', 'minimize'])
parser.add_argument('--n_trials', dest = 'n_trials', default = 100, type = int)
parser.add_argument('--timeout', dest = 'timeout', default = 600, type = int)
parser.add_argument('--feature_importance_file', dest = 'feature_importance_file')
args = parser.parse_args()

X = pd.read_csv(args.table_file, sep = '\t', quoting = 1, index_col = 0)
y = X[args.y_feature].tolist()
X = X.drop(args.y_feature, axis = 1)

print('X.shape', X.shape)
print('len(y)', len(y))

trial_number_model = {}

def objective(trial: optuna.Trial) -> float:
	random_state = int(datetime.datetime.utcnow().timestamp() * 1000000) % (2 ** 32)
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size = args.test_size,
		shuffle = True,
		stratify = y,
		random_state = random_state,
	)
	sm = SMOTE(random_state = random_state) # Synthetic Minority Over-sampling Technique
	X_train, y_train = sm.fit_resample(X_train, y_train)
	params = {
		'iterations': trial.suggest_categorical('iterations', [1000, 2000]), # = num_boost_round, n_estimators, num_trees
		'learning_rate': trial.suggest_float('learning_rate', 0.01, 1), # The smaller the value, the more iterations are required for training.
		'depth': trial.suggest_int('depth', 1, 12), # Depth of the tree. Decreasing the value prevents overfitting.
		'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 12), # = reg_lambda. Increasing the value prevents overfitting.
	}
	model = CatBoostClassifier(
		**params,
		eval_metric = args.eval_metric,
		custom_metric = args.custom_metric, # https://catboost.ai/en/docs/references/custom-metric__supported-metrics
		thread_count = args.thread_count,
		used_ram_limit = args.used_ram_limit,
		random_state = random_state,
	)
	pruning_callback = optuna.integration.CatBoostPruningCallback(trial, args.eval_metric)
	model.fit(
		X = X_train,
		y = y_train,
		cat_features = args.cat_feature,
		eval_set = (X_test, y_test),
		early_stopping_rounds = 100,
		callbacks = [pruning_callback],
		verbose = False,
	)
	pruning_callback.check_pruned() # evoke pruning manually.
	trial_number_model[trial.number] = model
	return model.get_best_score()['validation'][args.eval_metric]

study = optuna.create_study(
	pruner = optuna.pruners.MedianPruner(n_warmup_steps = 5),
	direction = args.study_direction,
)
study.optimize(
	objective,
	n_trials = args.n_trials,
	timeout = args.timeout,
)
print('len(trial_number_model)', len(trial_number_model))
print('len(study.trials)', len(study.trials))
print('study.best_value', study.best_value)

model = trial_number_model[study.best_trial.number]

print(model.get_params())

for pool in sorted(model.get_best_score().keys()):
	for metric in sorted(model.get_best_score()[pool].keys()):
		print(pool, metric, model.get_best_score()[pool][metric])

dump(model, args.model_file)

if args.feature_importance_file != None:
	pd.DataFrame(zip(model.feature_names_, model.feature_importances_), columns = ['name', 'importance']).to_csv(args.feature_importance_file, sep = '\t', index = False)
