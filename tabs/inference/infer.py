import os
import sys
import logging
import shutil
import warnings
import torch
import numpy as np
import gradio as gr
from time import sleep
from random import shuffle
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()
load_dotenv("sha256.env")

# Установка пути для временных файлов
now_dir = os.getcwd()
sys.path.append(now_dir)

# Настройки для macOS
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Импорт необходимых модулей
from infer.modules.vc import VC
from infer.lib.train.process_ckpt import change_info, extract_small_model, merge
from i18n.i18n import I18nAuto
from configs import Config

# Установка уровня логирования для сторонних библиотек
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Настройка логирования
logger = logging.getLogger(__name__)

# Создание временных и необходимых директорий
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp

# Отключение предупреждений
warnings.filterwarnings("ignore")

# Инициализация генератора случайных чисел для PyTorch
torch.manual_seed(114514)

# Загрузка конфигурации и инициализация голосового преобразователя (VC)
config = Config()
vc = VC(config)

# Обходная функция для DML (если включено)
if config.dml:
    from fairseq.modules.grad_multiply import GradMultiply

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        return x.clone().detach()

    GradMultiply.forward = forward_dml

# Инициализация многоязычной поддержки
i18n = I18nAuto()

# Получение информации о GPU
ngpu = torch.cuda.device_count()
gpu_infos, mem = [], []
if_gpu_ok = False

# Получение путей к активам из переменных окружения
weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
outside_index_root = os.getenv("outside_index_root")

# Получение всех весовых файлов
names = [name for name in os.listdir(weight_root) if name.endswith(".pth")]
index_paths = []

# Функция для поиска индексов
def lookup_indices(root_path):
    global index_paths
    for root, _, files in os.walk(root_path, topdown=False):
        index_paths.extend(f"{root}/{name}" for name in files if name.endswith(".index") and "trained" not in name)

lookup_indices(index_root)
lookup_indices(outside_index_root)

# Функция для обновления выбора весов и индексов
def change_choices():
    names = [name for name in os.listdir(weight_root) if name.endswith(".pth")]
    index_paths = []
    lookup_indices(index_root)
    return {"choices": sorted(names), "__type__": "update"}, {"choices": sorted(index_paths), "__type__": "update"}

# Функция для очистки выбора
def clean():
    return {"value": "", "__type__": "update"}
