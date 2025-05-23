from configparser import ConfigParser


def load_config(config_path: str = "config.ini") -> ConfigParser:
    """
    Loads configuration settings from the given .ini file.

    Args:
        config_path (str): Path to the config.ini file.

    Returns:
        ConfigParser: ConfigParser object containing configuration values.
    """
    config = ConfigParser()
    config.read(config_path)
    return config
