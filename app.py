from i18n.i18n import I18nAuto
import gradio as gr
from configs.config import Config
from infer.modules.vc.modules import VC
from tabs.infer import infer_tab
from tabs.train import train_tab


i18n = I18nAuto()
config = Config()
vc = VC(config)


with gr.Blocks(title="Kanoyo (BETA)") as app:
    gr.Markdown("## Kanoyo (BETA)")
    gr.Markdown(
        value=i18n(
            "This is test code, it may not work and may generate errors."
        )
    )
    with gr.Tab(i18n("Infer")):
        infer_tab()
        
    with gr.Tab(i18n("Train")):
        train_tab()

    if config.iscolab:
        app.queue(max_size=1022).launch(max_threads=511, share=True)
    else:
        app.queue(max_size=1022).launch(
            max_threads=511,
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
