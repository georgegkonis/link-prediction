import logging
import os


def setup_logging(name: str) -> logging.Logger:
    os.makedirs('logs', exist_ok=True)
    fmt = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    if not log.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(f'logs/{name}.log')
        fh.setFormatter(fmt)
        log.addHandler(sh)
        log.addHandler(fh)
    return log
