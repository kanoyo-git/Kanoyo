import os
import sys
from dotenv import load_dotenv
import logging
import shutil
import warnings
import torch
import platform
from infer.modules.vc import VC, show_info, hash_similarity
from infer.lib.train.process_ckpt import change_info, extract_small_model, merge
from i18n.i18n import I18nAuto
from configs import Config
from sklearn.cluster import MiniBatchKMeans
import faiss
import numpy as np
import traceback
from time import sleep
from random import shuffle
from subprocess import Popen
import threading
import pathlib
import json

# Настройка логирования
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
load_dotenv("sha256.env")

# Настройка окружения для macOS
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Инициализация конфигурации и модулей
config = Config()
vc = VC(config)
i18n = I18nAuto()
logger.info(i18n)

# Очистка и создание временных директорий
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

# Проверка и загрузка необходимых ресурсов
if not config.nocheck:
    from infer.lib.rvcmd import check_all_assets, download_all_assets
    if not check_all_assets(update=config.update):
        if config.update:
            download_all_assets(tmpdir=tmp)
            if not check_all_assets(update=config.update):
                logging.error("Could not satisfy all assets needed.")
                exit(1)

# Настройка для DML (DirectML)
if config.dml:
    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        return x.clone().detach()
    import fairseq
    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

# Получение информации о GPU
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

# Загрузка переменных окружения
weight_root = os.getenv("weight_root")
index_root = os.getenv("index_root")
outside_index_root = os.getenv("outside_index_root")

# Получение списка моделей
names = [name for name in os.listdir(weight_root) if name.endswith(".pth")]
index_paths = []

def transfer_files(filething, dataset_dir='assets/dataset/'):
    file_names = [f.name for f in filething]
    for f in file_names:
        filename = os.path.basename(f)
        destination = os.path.join(dataset_dir, filename)
        shutil.copyfile(f, destination)
    return i18n("Transferred files to dataset directory!")

# Функция для поиска индексов
def lookup_indices(index_root):
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append(f"{root}/{name}")

# Поиск индексов
lookup_indices(index_root)
lookup_indices(outside_index_root)

# Словарь для частот дискретизации
sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}

# Проверка доступности GPU
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper() for value in ["10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", "A50", "500", "A60", "70", "80", "90", "M4", "T4", "TITAN", "4060", "L", "6000"]):
            if_gpu_ok = True
            gpu_infos.append(f"{i}\t{gpu_name}")
            mem.append(int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4))

# Вывод информации о GPU
gpu_info = "\n".join(gpu_infos) if if_gpu_ok and gpu_infos else i18n("Unfortunately, there is no compatible GPU available to support your training.")
default_batch_size = min(mem) // 2 if if_gpu_ok else 1
gpus = "-".join([i[0] for i in gpu_infos])

# Флаг видимости GPU для F0
F0GPUVisible = not config.dml

# Функция для изменения метода F0
def change_f0_method(f0method8):
    visible = F0GPUVisible if f0method8 == "rmvpe_gpu" else False
    return {"visible": visible, "__type__": "update"}

# Функция для ожидания завершения процесса
def if_done(done, p):
    while p.poll() is None:
        sleep(0.5)
    done[0] = True

# Функция для ожидания завершения нескольких процессов

def transfer_files(filething, dataset_dir='dataset/'):
    file_names = [f.name for f in filething]
    for f in file_names:
        filename = os.path.basename(f)
        destination = os.path.join(dataset_dir, filename)
        shutil.copyfile(f, destination)
    return i18n("Transferred files to dataset directory!")

def if_done_multi(done, ps):
    while any(p.poll() is None for p in ps):
        sleep(0.5)
    done[0] = True

# Функция для получения предварительно обученных моделей
def get_pretrained_models(path_str, f0_str, sr2):
    g_path = f"assets/pretrained{path_str}/{f0_str}G{sr2}.pth"
    d_path = f"assets/pretrained{path_str}/{f0_str}D{sr2}.pth"
    return (g_path if os.access(g_path, os.F_OK) else "", d_path if os.access(d_path, os.F_OK) else "")

# Функция для изменения частоты дискретизации
def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)

# Функция для изменения версии модели
def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    choices = ["40k", "48k"] if version19 == "v1" else ["40k", "48k", "32k"]
    return (*get_pretrained_models(path_str, "f0" if if_f0_3 else "", sr2), {"choices": choices, "__type__": "update", "value": sr2})

# Функция для изменения F0
def change_f0(if_f0_3, sr2, version19):
    path_str = "" if version19 == "v1" else "_v2"
    return ({"visible": if_f0_3, "__type__": "update"}, {"visible": if_f0_3, "__type__": "update"}, *get_pretrained_models(path_str, "f0" if if_f0_3 else "", sr2))

# Функция для извлечения F0 и признаков
def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    gpus = gpus.split("-")
    os.makedirs(f"{now_dir}/logs/{exp_dir}", exist_ok=True)
    log_path = f"{now_dir}/logs/{exp_dir}/extract_f0_feature.log"
    open(log_path, "w").close()

    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = f'"{config.python_cmd}" infer/modules/train/extract/extract_f0_print.py "{now_dir}/logs/{exp_dir}" {n_p} {f0method}'
            logger.info("Execute: " + cmd)
            p = Popen(cmd, shell=True, cwd=now_dir)
            done = [False]
            threading.Thread(target=if_done, args=(done, p)).start()
        else:
            gpus_rmvpe = gpus_rmvpe.split("-")
            leng = len(gpus_rmvpe)
            ps = []
            for idx, n_g in enumerate(gpus_rmvpe):
                cmd = f'"{config.python_cmd}" infer/modules/train/extract/extract_f0_rmvpe.py {leng} {idx} {n_g} "{now_dir}/logs/{exp_dir}" {config.is_half}'
                logger.info("Execute: " + cmd)
                p = Popen(cmd, shell=True, cwd=now_dir)
                ps.append(p)
            done = [False]
            threading.Thread(target=if_done_multi, args=(done, ps)).start()
        while not done[0]:
            with open(log_path, "r") as f:
                yield f.read()
            sleep(1)
        with open(log_path, "r") as f:
            log = f.read()
        logger.info(log)
        yield log

    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = f'"{config.python_cmd}" infer/modules/train/extract_feature_print.py {config.device} {leng} {idx} {n_g} "{now_dir}/logs/{exp_dir}" {version19} {config.is_half}'
        logger.info("Execute: " + cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        ps.append(p)
    done = [False]
    threading.Thread(target=if_done_multi, args=(done, ps)).start()
    while not done[0]:
        with open(log_path, "r") as f:
            yield f.read()
        sleep(1)
    with open(log_path, "r") as f:
        log = f.read()
    logger.info(log)
    yield log

# Функция для предварительной обработки датасета
def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs(f"{now_dir}/logs/{exp_dir}", exist_ok=True)
    log_path = f"{now_dir}/logs/{exp_dir}/preprocess.log"
    open(log_path, "w").close()
    cmd = f'"{config.python_cmd}" infer/modules/train/preprocess.py "{trainset_dir}" {sr} {n_p} "{now_dir}/logs/{exp_dir}" {config.noparallel} {config.preprocess_per:.1f}'
    logger.info("Execute: " + cmd)
    p = Popen(cmd, shell=True)
    done = [False]
    threading.Thread(target=if_done, args=(done, p)).start()
    while not done[0]:
        with open(log_path, "r") as f:
            yield f.read()
        sleep(1)
    with open(log_path, "r") as f:
        log = f.read()
    logger.info(log)
    yield log

# Функция для запуска обучения
def click_train(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19, author):
    exp_dir = f"{now_dir}/logs/{exp_dir1}"
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = f"{exp_dir}/0_gt_wavs"
    feature_dir = f"{exp_dir}/3_feature256" if version19 == "v1" else f"{exp_dir}/3_feature768"
    f0_dir = f"{exp_dir}/2a_f0"
    f0nsf_dir = f"{exp_dir}/2b-f0nsf"
    names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set([name.split(".")[0] for name in os.listdir(feature_dir)])
    if if_f0_3:
        names &= set([name.split(".")[0] for name in os.listdir(f0_dir)]) & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{spk_id5}")
        else:
            opt.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{spk_id5}")
    fea_dim = 256 if version19 == "v1" else 768
    for _ in range(2):
        opt.append(f"{now_dir}/logs/mute/0_gt_wavs/mute{sr2}.wav|{now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|{now_dir}/logs/mute/2a_f0/mute.wav.npy|{now_dir}/logs/mute/2b-f0nsf/mute.wav.npy|{spk_id5}" if if_f0_3 else f"{now_dir}/logs/mute/0_gt_wavs/mute{sr2}.wav|{now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|{spk_id5}")
    shuffle(opt)
    with open(f"{exp_dir}/filelist.txt", "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    logger.info(f"Use gpus: {gpus16}")
    config_path = f"v1/{sr2}.json" if version19 == "v1" or sr2 == "40k" else f"v2/{sr2}.json"
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config.json_config[config_path], f, ensure_ascii=False, indent=4, sort_keys=True)
            f.write("\n")
    cmd = f'"{config.python_cmd}" infer/modules/train/train.py -e "{exp_dir1}" -sr {sr2} -f0 {int(if_f0_3)} -bs {batch_size12} -te {total_epoch11} -se {save_epoch10} {"-pg" if pretrained_G14 else ""} "{pretrained_G14}" {"-pd" if pretrained_D15 else ""} "{pretrained_D15}" -l {int(if_save_latest13 == i18n("Yes"))} -c {int(if_cache_gpu17 == i18n("Yes"))} -sw {int(if_save_every_weights18 == i18n("Yes"))} -v {version19} -a "{author}"'
    if gpus16:
        cmd += f' -g "{gpus16}"'
    logger.info("Execute: " + cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "Training complete. You can check the training logs in the console or the 'train.log' file under the experiment folder."

# Функция для обучения индекса
def train_index(exp_dir1, version19):
    exp_dir = f"logs/{exp_dir1}"
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = f"{exp_dir}/3_feature256" if version19 == "v1" else f"{exp_dir}/3_feature768"
    if not os.path.exists(feature_dir):
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    infos = []
    npys = [np.load(f"{feature_dir}/{name}") for name in sorted(listdir_res)]
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append(f"Trying doing kmeans {big_npy.shape[0]} shape to 10k centers.")
        yield "\n".join(infos)
        try:
            big_npy = MiniBatchKMeans(n_clusters=10000, verbose=True, batch_size=256 * config.n_cpu, compute_labels=False, init="random").fit(big_npy).cluster_centers_
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)
    np.save(f"{exp_dir}/total_fea.npy", big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append(f"{big_npy.shape},{n_ivf}")
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(index, f"{exp_dir}/trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index")
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i:i + batch_size_add])
    index_save_path = f"{exp_dir}/added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index"
    faiss.write_index(index, index_save_path)
    infos.append(i18n("Successfully built index into") + " " + index_save_path)
    link_target = f"{outside_index_root}/{exp_dir1}_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index"
    try:
        link = os.link if platform.system() == "Windows" else os.symlink
        link(index_save_path, link_target)
        infos.append(i18n("Link index to outside folder") + " " + link_target)
    except:
        infos.append(i18n("Link index to outside folder") + " " + link_target + " " + i18n("Fail"))
    yield "\n".join(infos)

# Функция для однокнопочного обучения
def train1key(exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19, gpus_rmvpe, author):
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    yield get_info_str(i18n("Step 1: Processing data"))
    [get_info_str(_) for _ in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7)]

    yield get_info_str(i18n("step2:Pitch extraction & feature extraction"))
    [get_info_str(_) for _ in extract_f0_feature(gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe)]

    yield get_info_str(i18n("Step 3a: Model training started"))
    click_train(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19, author)
    yield get_info_str(i18n("Training complete. You can check the training logs in the console or the 'train.log' file under the experiment folder."))

    [get_info_str(_) for _ in train_index(exp_dir1, version19)]
    yield get_info_str(i18n("All processes have been completed!"))

# Функция для изменения информации
def change_info_(ckpt_path):
    log_path = ckpt_path.replace(os.path.basename(ckpt_path), "train.log")
    if not os.path.exists(log_path):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(log_path, "r") as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if "version" in info and info["version"] == "v2" else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}