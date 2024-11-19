# This file is used to load model
import json
import os
import datetime

from models import create_model
from utils import logger


def log_config(args, path):
    log_path = os.path.join("logs", path)
    logger.configure(dir_log=log_path, format_strs=["stdout", "log"], timestamp=False)
    logger.log("args: {}".format(args))


def load_config(config_path="./model/config.json"):
    with open(config_path, "r") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = load_config()
    name = f"{datetime.datetime.now().strftime('%m-%d')}-w-1--all"
    log_config(config, name)
    for i in range(len(config)):
        model_config = config[i]
        model = create_model(model_config)
        model()
