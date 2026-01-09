# cSpell:ignore tigeropen
from tigeropen.common.consts import (Language,        # Language
                                     Market,          # Market
                                     BarPeriod,       # K-line period
                                     QuoteRight)      # Adjustment type
from tigeropen.tiger_open_config import TigerOpenClientConfig

def get_client_config():
    """
    https://quant.itigerup.com/#developer Retrieve developer information
    """
    client_config = TigerOpenClientConfig(props_path='/props/')
    return client_config

# Create the client_config object using the above-defined function
client_config = get_client_config()