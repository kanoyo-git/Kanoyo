import os
import sys
import gradio as gr
from infer.modules.vc.modules import VC
from i18n.i18n import I18nAuto
from configs.config import Config

i18n = I18nAuto()
config = Config()
vc = VC(config)

now_dir = os.getcwd()
sys.path.append(now_dir)

weight_root = os.getenv("weight_root")
index_root = os.getenv("index_root")

def clean():
    return {"value": "", "__type__": "update"}

def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    
    index_paths = []
    if index_root is not None:
        for root, dirs, files in os.walk(index_root, topdown=False):
            for name in files:
                if name.endswith(".index") and "trained" not in name:
                    index_paths.append("%s/%s" % (root, name))
    
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []


def infer_tab():
    with gr.Column():
        sid0 = gr.Dropdown(label=i18n("Модель"), choices=sorted(names))
        refresh_button = gr.Button(i18n("Обновить модели"), variant="primary")
        clean_button = gr.Button(i18n("Выгрузить модель"), variant="primary")
    spk_item = gr.Slider(
        minimum=0,
        maximum=2333,
        step=1,
        label=i18n("ID голоса"),
        value=0,
        visible=False,
        interactive=True,
    )
    clean_button.click(fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean")
    with gr.TabItem(i18n("Изменение голоса")):
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    vc_transform0 = gr.Number(
                        label=i18n("Питч"),
                        value=0,
                    )
                    input_audio0 = gr.Textbox(
                        label=i18n("Укажите путь к аудио"),
                        placeholder="C:\\Users\\Desktop\\audio_example.wav",
                    )
                    file_index1 = gr.Textbox(
                        label=i18n("Укажите путь к индексу"),
                        placeholder="C:\\Users\\Desktop\\model_example.index",
                        interactive=True,
                    )
                    file_index2 = gr.Dropdown(
                        label=i18n("Выберите индекс из выпадающего списка"),
                        choices=sorted(index_paths),
                        interactive=True,
                    )
                f0method0 = gr.Radio(
                    label=i18n(
                        "Выберите алгоритм извлечения тона. Быстрые но плохие по качеству - pm и dio, Неплохой - harvest, а лучший - rmvpe"
                    ),
                    choices=(
                        ["pm", "harvest", "crepe", "rmvpe"]
                        if config.dml == False
                        else ["pm", "harvest", "rmvpe"]
                    ),
                    value="rmvpe",
                    interactive=True,
                )

            with gr.Column():
                resample_sr0 = gr.Slider(
                    minimum=0,
                    maximum=48000,
                    label=i18n("Повторная дискретизация после обработки"),
                    value=0,
                    step=1,
                    interactive=True,
                )
                rms_mix_rate0 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n(
                        "Громкость исходного файла (чем ближе к 1 тем ближе к оригиналу)"
                    ),
                    value=0.25,
                    interactive=True,
                )
                protect0 = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n(
                        "Защита четких согласных и дыхательных звуков"
                    ),
                    value=0.33,
                    step=0.01,
                    interactive=True,
                )
                filter_radius0 = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label=i18n(
                        "Использование медианного фильтра"
                    ),
                    value=3,
                    step=1,
                    interactive=True,
                )
                index_rate1 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Влияние индекса"),
                    value=0.75,
                    interactive=True,
                )
                f0_file = gr.File(
                    label=i18n("Файл кривой F0"),
                    visible=False,
                )

                refresh_button.click(
                    fn=change_choices,
                    inputs=[],
                    outputs=[sid0, file_index2],
                    api_name="infer_refresh",
                )
                # file_big_npy1 = gr.Textbox(
                #     label=i18n("特征文件路径"),
                #     value="E:\\codes\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                #     interactive=True,
                # )