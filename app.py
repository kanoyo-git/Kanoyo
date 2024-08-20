import gradio as gr
from tabs.inference.infer_tab import infer_tab
from tabs.training.train_tab import train_tab
from i18n.i18n import I18nAuto
from configs import Config
i18n = I18nAuto()
config = Config()


with gr.Blocks(title="RVC WebUI") as app:
    gr.Markdown("## RVC WebUI")
    gr.Markdown(
        value=i18n(
            "This software is open source under the MIT license. The author does not have any control over the software. Users who use the software and distribute the sounds exported by the software are solely responsible. <br>If you do not agree with this clause, you cannot use or reference any codes and files within the software package. See the root directory <b>Agreement-LICENSE.txt</b> for details."
        )
    )
    with gr.Tabs():
        with gr.Tab(i18n("Infer")):
            infer_tab()

        with gr.Tab(i18n("Train")):
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
