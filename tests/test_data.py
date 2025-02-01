import os

def test_raw_data_exists():
    """
    Check if the raw data folder has files.
    """
    raw_data_dir = os.path.join("data", "raw")
    assert len(os.listdir(raw_data_dir)) > 0, "Raw data folder is empty!"
