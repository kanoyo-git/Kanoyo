import os
from dotenv import load_dotenv
from infer.modules.vc.modules import VC
from infer.lib.train.process_ckpt import change_info, extract_small_model, merge, show_info
from i18n.i18n import I18nAuto
from configs.config import Config
import torch
import numpy as np
import warnings
import logging
import gradio as gr
import sys

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

config = Config()
vc = VC(config)

i18n = I18nAuto()

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper() for value in ["10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", "A50", "500", "A60", "70", "80", "90", "M4", "T4", "TITAN", "4060", "L", "6000"]):
            if_gpu_ok = True
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4))

if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])
from gradio.events import Dependency

class ToolButton(gr.Button, gr.components.FormComponent):
    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"

weight_root = os.getenv("weight_root")
index_root = os.getenv("index_root")
outside_index_root = os.getenv("outside_index_root")

names = [name for name in os.listdir(weight_root) if name.endswith(".pth")]
index_paths = []

def lookup_indices(index_root):
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))

lookup_indices(index_root)
lookup_indices(outside_index_root)

def change_choices():
    names = [name for name in os.listdir(weight_root) if name.endswith(".pth")]
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {"choices": sorted(index_paths), "__type__": "update"}

def clean():
    return {"value": "", "__type__": "update"}