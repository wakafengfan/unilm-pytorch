import logging
import os
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict


ROOT_PATH = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

# data
data_dir = Path(ROOT_PATH)/ "data"
model_dir = Path(ROOT_PATH) / "model"

bert_data_path = Path.home() / 'db__pytorch_pretrained_bert'
bert_model_path = bert_data_path / 'bert-base-chinese'
roberta_model_path = bert_data_path / 'chinese_Roberta_bert_wwm_large_ext_pytorch'
bert_wwm_pt_path = bert_data_path / "chinese_wwm_ext_pytorch"
mt5_pt_path = bert_data_path / "mt5_small_pt"
nezha_pt_path = bert_data_path / "nezha-cn-base"
simbert_pt_path = bert_data_path / "chinese_simbert_pt"

common_data_path = Path.home() / 'db__common_dataset'
open_dataset_path = common_data_path / "open_dataset"
mrc_datset_path = open_dataset_path / "mrc"
dureader_dataset_path = open_dataset_path / "dureader"




###############################################
# log
###############################################

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f'begin progress ...')


