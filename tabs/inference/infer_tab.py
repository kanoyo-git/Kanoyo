import os
import sys
from .infer import *

now_dir = os.getcwd()
sys.path.append(now_dir)

def infer_tab():
    with gr.Blocks():
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                with gr.Accordion("Основные настройки", open=True):
                    vc_transform0 = gr.Slider(
                        minimum=-24,
                        maximum=24,
                        step=1,
                        label=i18n("Питч (в полутонах)"),
                        value=0,
                        interactive=True,
                    )
                    input_audio0 = gr.Audio(
                        label=i18n("Аудио файл"),
                        type="filepath",
                    )
                with gr.Accordion("Дополнительные настройки", open=False):
                    with gr.Row():
                        with gr.Column(scale=2):
                            resample_sr0 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label=i18n("Частота пересэмплирования"),
                                value=0,
                                step=1,
                                interactive=True,
                            )
                        with gr.Column(scale=2):
                            rms_mix_rate0 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n("Смешивание RMS"),
                                value=0.25,
                                interactive=True,
                            )
                            protect0 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label=i18n("Защита согласных"),
                                value=0.33,
                                step=0.01,
                                interactive=True,
                            )
                        with gr.Column(scale=2):
                            filter_radius0 = gr.Slider(
                                minimum=0,
                                maximum=7,
                                label=i18n("Радиус фильтра"),
                                value=3,
                                step=1,
                                interactive=True,
                            )
                            index_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n("Процент индекса"),
                                value=0.75,
                                interactive=True,
                            )
                            f0_file = gr.File(
                                label=i18n("Файл F0 (опционально)"),
                                visible=False
                            )
                            
            with gr.Column(scale=1):
                refresh_button = gr.Button(
                    i18n("Обновить"), variant="primary"
                )
                clean_button = gr.Button(
                    i18n("Очистить"), variant="primary"
                )
                sid0 = gr.Dropdown(
                    label=i18n("Модель голоса"),
                    choices=sorted(names),
                    interactive=True,
                )
                file_index1 = gr.Dropdown(
                    label=i18n("Путь к индексу (авто)"),
                    choices=sorted(index_paths),
                    interactive=True,
                )
                f0method0 = gr.Radio(
                    label=i18n("Метод извлечения F0"),
                    choices=(
                        ["harvest", "crepe", "rmvpe", "fcpe"]
                        if config.dml == False
                        else ["harvest", "rmvpe"]
                    ),
                    value="rmvpe",
                    interactive=True,
                )
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("ID говорящего"),
                    value=0,
                    interactive=True,
                    visible=False,
                )
                but0 = gr.Button(i18n("Конвертировать"), variant="primary", scale=1)
            
        with gr.Row():
            vc_output1 = gr.Textbox(label=i18n("Консоль"), interactive=False)
            vc_output2 = gr.Audio(
                label=i18n("Аудио")
            )
        refresh_button.click(
            fn=change_choices,
            inputs=[],
            outputs=[sid0, file_index1],
            api_name="infer_refresh",
        )
        clean_button.click(
            fn=clean,
            inputs=[],
            outputs=[sid0],
            api_name="infer_clean"
        )
        but0.click(
            vc.vc_single,
            [
                spk_item,
                input_audio0,
                vc_transform0,
                f0_file,
                f0method0,
                file_index1,
                file_index1,
                index_rate1,
                filter_radius0,
                resample_sr0,
                rms_mix_rate0,
                protect0,
            ],
            [vc_output1, vc_output2],
            api_name="infer_convert",
        )
        sid0.change(
            fn=vc.get_vc,
            inputs=[sid0, protect0, file_index1],
            outputs=[protect0, file_index1],
            api_name="infer_change_voice",
        )

infer_tab()