import json
import os
from pathlib import Path
from typing import Dict

from nas_unzip.nas import nas_unzip


def get_creds() -> Dict[str, str]:
    """Obtains the credentials

    Returns:
        Dict[str, str]: Username and password dictionary
    """
    if Path("credentials.json").is_file():
        with open("credentials.json", "r", encoding="ascii") as handle:
            return json.load(handle)
    else:
        value = os.environ["NAS_CREDS"].splitlines()
        assert len(value) == 2
        return {"username": value[0], "password": value[1]}


def download_data(creds: Dict[str, str]):
    script_path = Path(__file__)

    print(script_path.parent / "data")

    nas_unzip(
        network_path="smb://e4e-nas.ucsd.edu:6021/temp/github_actions/pyFishSenseDev/pyFishSenseDevTest.zip",
        output_path=script_path.parent / "data",
        username=creds["username"],
        password=creds["password"],
    )


if __name__ == "__main__":
    creds = get_creds()
    download_data(creds)
