from .models import *
import gradio as gr


def models_tab():
        with gr.TabItem(i18n("Download Voice Models")):
             gr.Markdown(i18n("For models found in AI Hub"))
        with gr.Row():
            url = gr.Textbox(label=i18n("Huggingface Link:"))
        with gr.Row():
            model = gr.Textbox(label=i18n("Name of the model (without spaces):"))
            download_button = gr.Button(i18n("Download"))
        with gr.Row():
            status_bar = gr.Textbox(label=i18n("Download Status"))
        download_button.click(fn=download_from_url, inputs=[url, model], outputs=[status_bar])

        with gr.TabItem(i18n("Import Models")):
            gr.Markdown(i18n("For models found on Weights"))
            file_upload = gr.File(label=i18n("Upload a .zip file containing a .pth and .index file"))
            import_button = gr.Button(i18n("Import"))
            import_status = gr.Textbox(label=i18n("Import Status"))
            import_button.click(fn=import_button_click, inputs=file_upload, outputs=import_status)