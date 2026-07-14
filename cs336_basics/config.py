import logging
from pathlib import Path

class LevelFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG:      logging.Formatter("%(relativeCreated)08d ms - %(module)s:%(funcName)s:%(lineno)d %(message)s"),
        logging.INFO:       logging.Formatter("%(asctime)s %(relativeCreated)08d ms %(message)s"),
        logging.WARNING:    logging.Formatter("%(asctime)s %(relativeCreated)08d ms %(levelname)-8s- %(module)s:%(funcName)s:%(lineno)d %(message)s"),
    }

    def format(self, record):
        formatter = self.FORMATS.get(record.levelno, self.FORMATS.get(logging.WARNING))
        return formatter.format(record)

logger = logging.getLogger(__name__)

_configured_logging = False

def config_logging(log_level=logging.DEBUG):
    global _configured_logging

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if _configured_logging:
        logger.debug("Logging already configured")
        return

    console_h = logging.StreamHandler()
    console_h.setFormatter(LevelFormatter())
    root_logger.addHandler(console_h)

    file_h = logging.FileHandler("train.log")
    file_h.setFormatter(LevelFormatter())
    root_logger.addHandler(file_h)

    _configured_logging = True
    logger.debug("Configured logging")

def get_data_dir(path = ''):
    config_logging()
    data_dir = str(Path(__file__).parent.parent) + "/data/"
    logger.debug(data_dir)
    return data_dir
