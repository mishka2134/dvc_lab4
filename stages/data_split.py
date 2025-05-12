import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

if __name__ == "__main__":
    with open("./config.yaml") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv("./data/energy_cleaned.csv")
    X = df.drop(columns=[config['target']])
    y = df[config['target']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_state']
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv("./data/train.csv", index=False)
    test.to_csv("./data/test.csv", index=False)
