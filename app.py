import warnings
import shutil
import logging
import os
import sys
import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from i18n.i18n import I18nAuto
from configs.config import Config
from tabs.inference.infer_tab import infer_tab
from tabs.training.train_tab import train_tab

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")

i18n = I18nAuto()
config = Config()
logger.info(i18n)

with gr.Blocks(title="RVC WebUI") as app:
    gr.Markdown("## RVC WebUI")
    gr.Markdown(
        value=i18n(
            "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
        )
    )
    with gr.Tabs():
        with gr.Tab(i18n("Infer")):
            infer_tab()

        with gr.Tab(i18n("Train")):
            train_tab()

    if config.iscolab:
        app.launch(share=True, max_threads=511)
    else:
        app.launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
            max_threads=511,
        )
