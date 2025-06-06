import os
import logging
from datetime import datetime
from pathlib import Path

root_path = Path(str(os.path.join(os.path.abspath(os.path.dirname(__file__))))).absolute()
log_folder = os.path.join(root_path, 'resource', 'log')
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

log_file_name = os.path.join(log_folder, '{}_{}.txt'.format('log', datetime.now().strftime('%m%d%Y%H%M%S')))
format_ = "%(asctime)s %(process)d %(module)s %(lineno)d %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format=format_, filename=log_file_name)
console_logger = logging.StreamHandler()
# console output format
stream_format = logging.Formatter("%(asctime)s %(process)d %(module)s %(lineno)d %(message)s")
# file output format
console_logger.setFormatter(stream_format)
logger.addHandler(console_logger)

#以日志的形式存储结果
