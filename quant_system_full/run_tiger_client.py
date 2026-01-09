from tigeropen.common.consts import (
    Language,  # Language
    Market,    # Market
    BarPeriod, # K-line period
    QuoteRight # Adjustment type
)
from tigeropen.tiger_open_config import TigerOpenClientConfig
from pathlib import Path


def get_client_config():
    """
    https://quant.itigerup.com/#developer Retrieve developer information
    """
    props_dir = str((Path(__file__).parent / "props").resolve())
    print("Using props_path:", props_dir)
    client_config = TigerOpenClientConfig(props_path=props_dir)
    return client_config


# Create the client_config object using the above-defined function
client_config = get_client_config()
print("TigerOpenClientConfig loaded.")


