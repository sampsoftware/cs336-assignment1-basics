import logging
from pathlib import Path


class LevelFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: logging.Formatter(
            "%(relativeCreated)08d ms - %(mem)s - %(module)s:%(funcName)s:%(lineno)d %(message)s"
        ),
        logging.INFO: logging.Formatter("%(asctime)s %(relativeCreated)08d ms %(message)s"),
        logging.WARNING: logging.Formatter(
            "%(asctime)s %(relativeCreated)08d ms %(levelname)-8s- %(module)s:%(funcName)s:%(lineno)d %(message)s"
        ),
    }

    def format(self, record):
        formatter = self.FORMATS.get(record.levelno, self.FORMATS.get(logging.WARNING))
        return formatter.format(record)


logger = logging.getLogger(__name__)

_configured_logging = False

peak_mem = 0


def config_logging(log_level=logging.DEBUG):
    global _configured_logging

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if _configured_logging:
        logger.debug("Logging already configured")
        return

    console_h = logging.StreamHandler()
    console_h.setFormatter(LevelFormatter())
    console_h.addFilter(MemFilter())
    root_logger.addHandler(console_h)

    file_h = logging.FileHandler("train.log")
    file_h.setFormatter(LevelFormatter())
    file_h.addFilter(MemFilter())
    root_logger.addHandler(file_h)
    _configured_logging = True
    logger.debug("Configured logging")


def get_data_dir(path=""):
    config_logging()
    data_dir = str(Path(__file__).parent.parent) + "/data/"
    logger.debug(data_dir)
    return data_dir


class MemFilter(logging.Filter):
    def filter(self, record):
        b = _read_int(_MEM_PATH) if _MEM_PATH else None
        record.mem = f"{b / 2**30:.2f}GiB" if b is not None else "?"
        return True  # always let the record through


def _read_int(path):
    try:
        return int(Path(path).read_text())
    except (OSError, ValueError):  # missing, RO, permission, non-int -> give up quietly
        return None


def _resolve_mem_path():
    # v2, this container's own nested cgroup (e.g. /docker/465936…)
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().rsplit("::", 1)[-1]
        p = f"/sys/fs/cgroup{rel}/memory.current"
        if _read_int(p) is not None:
            return p
    except OSError:
        pass
    for p in (
        "/sys/fs/cgroup/memory.current",  # v2, namespaced-at-root
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ):  # v1
        if _read_int(p) is not None:
            return p
    return None


_MEM_PATH = _resolve_mem_path()  # module load, once
