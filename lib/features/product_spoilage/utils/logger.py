# utils/logger.py
from datetime import datetime
from threading import Thread

from lib.features.product_spoilage.config import config


def setup_logger():
    def async_log(message):
        def worker():
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            with open(config.LOG_FILE, "a") as f:
                f.write(f"[{timestamp}] {message}\n")
        Thread(target=worker).start()
    return async_log