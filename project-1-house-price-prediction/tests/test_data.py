import pandas as pd

from house_price.data.load import get_dataset, load_raw_data


def test_load_raw_data_reads_csv(tmp_path):
    data_dir = tmp_path
    df = pd.DataFrame({"SalePrice": [100000, 200000]})
    df.to_csv(data_dir / "train.csv", index=False)

    result = load_raw_data(filename="train.csv", data_dir=data_dir)

    assert list(result["SalePrice"]) == [100000, 200000]


def test_get_dataset_prefers_local_file(tmp_path):
    data_dir = tmp_path
    df = pd.DataFrame({"SalePrice": [123]})
    df.to_csv(data_dir / "train.csv", index=False)

    result = get_dataset(filename="train.csv", data_dir=data_dir)

    assert list(result["SalePrice"]) == [123]
