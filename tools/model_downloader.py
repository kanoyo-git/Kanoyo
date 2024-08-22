import os
import subprocess
import logging

logger = logging.getLogger(__name__)

URL_BASE = "https://huggingface.co/kanoyo/0v2Super/resolve/main"
models_download = [
    (
        "pretrained_v2/",
        [
            "f0D40k.pth",
            "f0G40k.pth",
            "G_SnowieV3.1_40k.pth",
            "D_SnowieV3.1_40k.pth",
            "G_Snowie-X-Rin_40k.pth",
            "D_Snowie-X-Rin_40k.pth",
            "G_SnowieV3.1_48k.pth",
            "D_SnowieV3.1_48k.pth",
            "G_Rigel_32k.pth",
            "D_Rigel_32k.pth",
        ],
    ),
]

individual_files = [
    ("hubert_base.pt", "assets/hubert/"),
    ("rmvpe.pt", "assets/rmvpe/"),
    ("rmvpe.onnx", "assets/rmvpe/"),
]

folder_mapping = {
    "pretrained_v2/": "assets/pretrained_v2/",
    "": "",
}

def download_file_with_aria2c(url, destination_path):
    command = [
        "aria2c",
        "--log-level=error",  
        "-x", "16",  
        "-s", "16",  
        "-k", "1M",  
        "-d", os.path.dirname(destination_path),
        "-o", os.path.basename(destination_path),
        url
    ]
    with open(os.devnull, 'w') as devnull:
        subprocess.run(command, stdout=devnull, stderr=devnull)

for remote_folder, file_list in models_download:
    local_folder = folder_mapping.get(remote_folder, "")
    for file in file_list:
        destination_path = os.path.join(local_folder, file)
        url = f"{URL_BASE}/{remote_folder}{file}"
        if not os.path.exists(destination_path):
            print(f"Скачивание {url} в {destination_path}...")
            download_file_with_aria2c(url, destination_path)

for file_name, local_folder in individual_files:
    destination_path = os.path.join(local_folder, file_name)
    url = f"{URL_BASE}/{file_name}"
    if not os.path.exists(destination_path):
        print(f"Скачивание {url} в {destination_path}...")
        download_file_with_aria2c(url, destination_path)

os.system("cls" if os.name == "nt" else "clear")
logger.info("Загрузка Kanoyo успешно продолжается...")
