import os
import pytest
from src.config_helper import load_settings
from src.ingestion.master_downloader import MasterDownloader

def test_master_downloader_init():
    downloader = MasterDownloader()
    assert downloader.download_dir == "data/master"
    assert downloader.enabled is True

def test_master_downloader_download_and_parse():
    downloader = MasterDownloader()
    # Force mock generation by executing download on index page
    y_txt_path = downloader.download()
    assert os.path.exists(y_txt_path)

    # Parse and verify records contain expected keys and at least one item
    records = downloader.parse_master(y_txt_path)
    assert len(records) > 0
    first_record = records[0]
    assert "yj_code" in first_record
    assert "japic_code" in first_record
    assert "name_ja" in first_record

    # Verify look up functionality
    target_yj = "4291043M1027"  # イムブルビカ
    drug = downloader.get_drug_by_yj_code(target_yj)
    assert drug != {}
    assert drug["yj_code"] == target_yj
    assert drug["japic_code"] == "4291043"
    assert "イムブルビカ" in drug["name_ja"]
