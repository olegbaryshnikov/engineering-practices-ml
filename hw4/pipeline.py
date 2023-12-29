from prefect import task, flow
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import requests

import neptune
from neptune.utils import stringify_unsupported

PROJECT_NAME = "olegbaryshnikov/engineering-practices-ml"
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzOTNkY2IzNS1kMDRjLTQ3MTAtODJiOS0wMjgzMjRkMjk0MzUifQ=="

@flow(log_prints=True)
def titanic_model(
        random_seed=1111,
        output_path="./output", 
        mid_res_path="./mid_results",
        download_url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        split_ratio=0.9, 
        model_depth=4, 
        model_learning_rate=0.05
    ):

    # start neptune run and save parameters
    run = init_neptune({
        "random_seed": random_seed,
        "split_ratio": split_ratio,
        "model_depth": model_depth,
        "model_learning_rate": model_learning_rate
    })

    # сбор/выгрузка данных из источника (например kaggle)
    data = load_dataset(download_url)

    # обработка данных (кодирование, шкалирование, заполнение пустот и тд)
    data = process_dataset(data)
    save_datasets([data], ["processed_data"], mid_res_path)

    # разбиение на train-val-test датасеты
    train, val, test = split_dataset(data, split_ratio, random_seed)
    save_datasets([train, val, test], ["train", "val", "test"], mid_res_path)

    train_X, train_y = split_data(train)
    val_X, val_y = split_data(val)
    
    # обучение/подбор гиперпараметров для модели
    model = train_model(train_X, train_y, val_X, val_y, model_depth, model_learning_rate, random_seed)

    # замер результатов модели на test выборке, оценка различных ошибок
    y_test = predict(model, test)
    y_test_proba = predict_proba(model, test)

    save_output(test, y_test, f"{output_path}/output.csv")

    accuracy = get_accuracy(test["Survived"], y_test)
    roc_auc = get_roc_auc_curve(test["Survived"], y_test_proba, f"{output_path}/roc_auc_curve.png")

    # use neptune to track data of run
    track_results(run, accuracy, roc_auc, model, f"{output_path}/output.csv")
    save_model(run, model)

    # finish neptune run
    dispose_run(run)

    return accuracy, roc_auc

@task(retries=2)
def load_dataset(url):
    res = requests.get(url, allow_redirects=True)
    with open('titanic.csv','wb') as file:
        file.write(res.content)
    
    return pd.read_csv("./titanic.csv")

@task
def save_datasets(datasets, names, path_to_save):
    for ds, name in zip(datasets, names):
        ds.to_csv(f"{path_to_save}/{name}.csv", index=False)


@task
def process_dataset(data):
    data = data.drop(["Age", "Cabin", "Name", "Ticket"], axis=1)
    data["Sex"] = data["Sex"].map({"male": 1, "female": 0}).astype(int)

    data["Embarked"].fillna("S", inplace=True)
    data["Embarked"] = data["Embarked"].map({"S": 1, "C": 2, "Q": 3}).astype(int)  

    return data


@task
def split_dataset(data, ratio, random_seed):
    train_val, test = train_test_split(data, train_size=ratio, random_state=random_seed, stratify=data["Survived"])
    train, val = train_test_split(train_val, train_size=ratio, random_state=random_seed, stratify=train_val["Survived"])

    return train, val, test

@task
def split_data(data):
    X, y = data.drop(columns=["Survived"]), data["Survived"]
    return X, y

@task
def train_model(train_X, train_y, val_X, val_y, depth, learning_rate, random_seed):
    model = CatBoostClassifier(
        iterations=500,
        early_stopping_rounds=100,
        depth=depth,
        learning_rate=learning_rate,
        eval_metric="Accuracy",
        random_seed=random_seed,
        allow_writing_files=False
        )

    model.fit(train_X, train_y, eval_set=(val_X, val_y))

    return model

@task
def predict(model, data):
    y = model.predict(data)
    return y

@task
def predict_proba(model, data):
    y_proba = model.predict_proba(data)[:,1]
    return y_proba

@task
def save_output(X, y, path_to_save):
    output = X.copy()
    output["Survived"] = y
    output.to_csv(path_to_save, index=False)
    return output

@task
def get_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Model accuracy: {accuracy}")
    return accuracy

@task
def get_roc_auc_curve(y_true, y_proba, path_to_save):
    fpr, tpr, threshold = roc_curve(y_true, y_proba)

    plt.plot(fpr, tpr)
    plt.savefig(path_to_save)

    return auc(fpr, tpr)

@task
def init_neptune(params):
    run = neptune.init_run(project=PROJECT_NAME, api_token=NEPTUNE_API_TOKEN)
    run["parameters"] = params

    return run

@task
def track_results(run, accuracy, roc_auc, model, output_csv):
    run["test/accuracy"] = accuracy
    run["test/roc_auc"] = roc_auc
    run["training/best_score"] = stringify_unsupported(model.get_best_score())
    run["training/best_iteration"] = stringify_unsupported(model.get_best_iteration())
    run["data/results"].upload(output_csv)

@task
def save_model(run, model):
    model.save_model("model.cbm")
    run["model/binary"].upload("model.cbm")
    run["model/attributes/feature_importances"] = dict(
        zip(model.feature_names_, model.get_feature_importance())
    )

@task
def dispose_run(run):
    run.stop()

if __name__ == "__main__":
    # start multiple runs with different parameters (grid search)
    for model_depth in [3,4,5]:
        for model_learning_rate in [0.01, 0.03, 0.05, 0.1]:
            titanic_model(model_depth=model_depth, model_learning_rate=model_learning_rate)