import gradio as gr
from tabs.inference.infer_tab import infer_tab
from tabs.training.train_tab import train_tab
from i18n.i18n import I18nAuto
from configs import Config
i18n = I18nAuto()
config = Config()


with gr.Blocks(title="Kanoyo") as app:
    gr.Markdown("## Kanoyo")
    gr.Markdown(
        value=i18n(
            "Преобразование голоса с открытым исходным кодом и огромным функционалом."
        )
    )
    with gr.Tabs():
        with gr.Tab(i18n("Изменить голос")):
            infer_tab()

        with gr.Tab(i18n("Обучение")):
            train_tab()

    if config.global_link:
        app.queue(max_size=1022).launch(share=True, max_threads=511)
    else:
        app.queue(max_size=1022).launch(
            max_threads=511,
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
