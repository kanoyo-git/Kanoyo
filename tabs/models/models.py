import os
import subprocess
import requests
from mega import Mega
from urllib.parse import urlencode
import shutil
import zipfile
import sys 

# Установка пути для временных файлов
now_dir = os.getcwd()
sys.path.append(now_dir)

from i18n.i18n import I18nAuto
i18n = I18nAuto()

def download_from_url(url, model):
    if url == '':
        return "Ссылка не может быть пустой."
    if model == '':
        return "Вы не написали имя модели. Пример: Kanoyo"

    url = url.strip()
    zip_dirs = ["zips", "unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)

    os.makedirs("zips", exist_ok=True)
    os.makedirs("unzips", exist_ok=True)

    zipfile = model + '.zip'
    zipfile_path = './zips/' + zipfile

    try:
        if "drive.google.com" in url:
            subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
        elif "mega.nz" in url:
            m = Mega()
            m.download_url(url, './zips')
        elif "disk.yandex.ru" in url:
            base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
            public_key = url
            final_url = base_url + urlencode(dict(public_key=public_key))
            response = requests.get(final_url)
            download_url = response.json()['href']
            download_response = requests.get(download_url)
            with open(zipfile_path, 'wb') as file:
                file.write(download_response.content)
        else:
            response = requests.get(url)
            response.raise_for_status() 
            with open(zipfile_path, 'wb') as file:
                file.write(response.content)

        shutil.unpack_archive(zipfile_path, "./unzips", 'zip')

        for root, dirs, files in os.walk('./unzips'):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".index"):
                    os.makedirs(f'./models/index', exist_ok=True)
                    shutil.copy2(file_path, f'./models/index/{model}.index')
                elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                    os.makedirs(f'./models/pth', exist_ok=True)
                    shutil.copy(file_path, f'./models/pth/{model}.pth')

        shutil.rmtree("zips")
        shutil.rmtree("unzips")
        return i18n("Модель загружена, вы можете вернуться на вкладку для изменения голоса!")

    except subprocess.CalledProcessError as e:
        return f"ERROR - Ошибка загрузки (gdown): {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"ERROR - Ошибка загрузки (requests): {str(e)}"
    except Exception as e:
        return f"ERROR - Тест провален: {str(e)}"
    
def import_files(file):
    if file is not None:
        file_name = file.name
        if file_name.endswith('.zip'):
            with zipfile.ZipFile(file.name, 'r') as zip_ref:
                # Create a temporary directory to extract files
                temp_dir = './TEMP'
                zip_ref.extractall(temp_dir)
                # Move .pth and .index files to their respective directories
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.pth'):
                            destination = './models/pth/' + file
                            if not os.path.exists(destination):
                                shutil.move(os.path.join(root, file), destination)
                            else:
                                print(f"Файл {destination} уже существует. Пропускаю.")
                        elif file.endswith('.index'):
                            destination = './models/index/' + file
                            if not os.path.exists(destination):
                                shutil.move(os.path.join(root, file), destination)
                            else:
                                print(f"Файл {destination} уже существует. Пропускаю.")
                # Remove the temporary directory
                shutil.rmtree(temp_dir)
            return "Зип-файл успешно распакован."
        elif file_name.endswith('.pth'):
            destination = './models/pth/' + os.path.basename(file.name)
            if not os.path.exists(destination):
                os.rename(file.name, destination)
            else:
                print(f"Файл {destination} уже существует. Пропускаю.")
            return "PTH файл успешно импортирован."
        elif file_name.endswith('.index'):
            destination = './models/index/' + os.path.basename(file.name)
            if not os.path.exists(destination):
                os.rename(file.name, destination)
            else:
                print(f"Файл {destination} уже существует. Пропускаю.")
            return "Файл индекса успешно импортирован."
        else:
            return "Неподдерживаемый тип файла."
    else:
        return "Файл не был загружен."
    
def import_button_click(file):
    return import_files(file)