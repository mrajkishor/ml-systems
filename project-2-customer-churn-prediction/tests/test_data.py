import pandas as pd

from churn_prediction.data.load import get_dataset, load_raw_data


def test_load_raw_data_reads_csv(tmp_path):
    df = pd.DataFrame({"Churn": ["Yes", "No"]})
    df.to_csv(tmp_path / "train.csv", index=False)

    result = load_raw_data(filename="train.csv", data_dir=tmp_path)

    assert list(result["Churn"]) == ["Yes", "No"]


def test_get_dataset_prefers_local_file(tmp_path):
    df = pd.DataFrame({"Churn": ["Yes"]})
    df.to_csv(tmp_path / "train.csv", index=False)

    result = get_dataset(filename="train.csv", data_dir=tmp_path)

    assert list(result["Churn"]) == ["Yes"]
