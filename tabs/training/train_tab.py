import gradio as gr
from .train import *

def train_tab():
    gr.Markdown(value=i18n(""))
    with gr.Row():
        exp_dir1 = gr.Textbox(label=i18n("Имя модели"), value="model-name")
        sr2 = gr.Radio(
            label=i18n("Частота дискретизации"),
            choices=["32k", "40k", "48k"],
            value="40k",
            interactive=True,
        )
        version19 = gr.Radio(
            label=i18n("V2 онли еее 42 братуха"),
            choices=["v2"],
            value="v2",
            interactive=False,
            visible=False,
        )
        np7 = gr.Slider(
            minimum=0,
            maximum=config.n_cpu,
            step=1,
            label=i18n("Потоки CPU"),
            value=int(np.ceil(config.n_cpu / 2.5)),
            interactive=True,
        )
    with gr.Group():
        gr.Markdown(value=i18n(""))
        with gr.Row():
            trainset_dir4 = gr.Textbox(
                label=i18n("Путь к датасету"), value="dataset"
            )
            with gr.Accordion(i18n('Загрузить датасет (альтернатива)'), open=False, visible=True):
                file_thin = gr.Files(label=i18n('Аудио файлы в формате wav, mp3, flac или ogg')) 
                show = gr.Textbox(label=i18n('Консоль'))
                transfer_button = gr.Button(i18n('Загрузить датасет'), variant="primary")
                transfer_button.click(
                    fn=transfer_files,
                    inputs=[file_thin],
                    outputs=show,
                )

    with gr.Group():
        gr.Markdown(value=i18n(""))
        with gr.Row():
            save_epoch10 = gr.Slider(
                minimum=1,
                maximum=250,
                step=1,
                label=i18n("Частота сохранения"),
                value=50,
                interactive=True,
            )
            total_epoch11 = gr.Slider(
                minimum=2,
                maximum=10000,
                step=1,
                label=i18n("Всего эпох"),
                value=300,
                interactive=True,
            )
            batch_size12 = gr.Slider(
                minimum=1,
                maximum=16,
                step=1,
                label=i18n("Батч-size"),
                value=default_batch_size,
                interactive=True,
            )
            if_save_every_weights18 = gr.Radio(
                label=i18n("Сохранять модель с выбранной частотой"),
                choices=[i18n("Да"), i18n("Нет")],
                value=i18n("Да"),
                interactive=True,
            )

    with gr.Accordion(i18n('Дополнительные настройки'), open=False, visible=True):
        with gr.Row(): 
            with gr.Group():
                spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label=i18n("ID голоса"),
                        value=0,
                        interactive=True,
                    )
                if_f0_3 = gr.Radio(
                label=i18n("Модель имеет поддержку работы с питчем"),
                choices=[True, False],
                value=True,
                interactive=True,
            )
                gpus6 = gr.Textbox(
                        label=i18n("ID ГПУ (Оставьте 0 если используете 1 GPU, используйте 0-1 для нескольких)"),
                        value=gpus,
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                gpu_info9 = gr.Textbox(
                        label=i18n("GPU Model"),
                        value=gpu_info,
                        visible=F0GPUVisible,
                    )
                gpus16 = gr.Textbox(
                label=i18n("ID ГПУ (Оставьте 0 если используете 1 GPU, используйте 0-1 для нескольких"),
                value=gpus if gpus != "" else "0",
                interactive=True,
                )
                with gr.Group():
                    if_save_latest13 = gr.Radio(
                        label=i18n("Сохранять последний ckpt как финальную модель"),
                        choices=[i18n("Да"), i18n("Нет")],
                        value=i18n("Да"),
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label=i18n("Кэшировать датасет в GPU (Только если датасет менее 8 минут)"),
                        choices=[i18n("Да"), i18n("Нет")],
                        value=i18n("Нет"),
                        interactive=True,
                    )
                    f0method8 = gr.Radio(
                            label=i18n("F0 метод"),
                            choices=["rmvpe", "rmvpe_gpu"],
                            value="rmvpe_gpu",
                            interactive=True,
                        )
                    gpus_rmvpe = gr.Textbox(
                            label=i18n(
                                "rmvpe_gpu будет использовать ваш GPU вместо CPU для извлечения функций."
                            ),
                            value="%s-%s" % (gpus, gpus),
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                    f0method8.change(
                        fn=change_f0_method,
                        inputs=[f0method8],
                        outputs=[gpus_rmvpe],
                    )        

    with gr.Row():
        pretrained_G14 = gr.Textbox(
            label=i18n("Претрейн G"),
            value="assets/pretrained_v2/f0G40k.pth",
            interactive=True,
        )
        pretrained_D15 = gr.Textbox(
            label=i18n("Претрейн D"),
            value="assets/pretrained_v2/f0D40k.pth",
            interactive=True,
        )
        sr2.change(
            change_sr2,
            [sr2, if_f0_3, version19],
            [pretrained_G14, pretrained_D15],
        )
        version19.change(
            change_version19,
            [sr2, if_f0_3, version19],
            [pretrained_G14, pretrained_D15, sr2],
        )
        if_f0_3.change(
            change_f0,
            [if_f0_3, sr2, version19],
            [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15],
        )
    
    with gr.Group():
        with gr.Row():
            but1 = gr.Button(i18n("1. Обработать данные"), variant="primary")
            but2 = gr.Button(i18n("2. Извлечь функции"), variant="primary")
            but4 = gr.Button(i18n("3. Обучить индекс"), variant="primary")
            but3 = gr.Button(i18n("4. Обучить модель"), variant="primary")
            info = gr.Textbox(label=i18n("Консоль"), value="", max_lines=5, lines=5)
            but1.click(
                preprocess_dataset,
                [trainset_dir4, exp_dir1, sr2, np7],
                [info],
                api_name="train_preprocess",
            )
            but2.click(
            extract_f0_feature,
                [
                    gpus6,
                    np7,
                    f0method8,
                    if_f0_3,
                    exp_dir1,
                    version19,
                    gpus_rmvpe,
                ],
                [info],
                api_name="train_extract_f0_feature",
            )
            but4.click(train_index, [exp_dir1, version19], info)
            but3.click(
            click_train,
            [
                exp_dir1,
                sr2,
                if_f0_3,
                spk_id5,
                save_epoch10,
                total_epoch11,
                batch_size12,
                if_save_latest13,
                pretrained_G14,
                pretrained_D15,
                gpus16,
                if_cache_gpu17,
                if_save_every_weights18,
                version19,
            ],
            info,
            api_name="train_start",
            )
            but4.click(train_index, [exp_dir1, version19], info)