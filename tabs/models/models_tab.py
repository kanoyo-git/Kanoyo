from .models import *
import gradio as gr


def models_tab():
        gr.Markdown(i18n("Для моделей найденных на просторах интернета."))
        with gr.Row():
            url = gr.Textbox(label=i18n("Ссылка:"))
        with gr.Row():
            model = gr.Textbox(label=i18n("Название модели (без пробелов):"))
            download_button = gr.Button(i18n("Скачать"))
        with gr.Row():
            status_bar = gr.Textbox(label=i18n("Консоль"))
        download_button.click(fn=download_from_url, inputs=[url, model], outputs=[status_bar])

        gr.Markdown(i18n("Для зип-архивов и локальных моделей"))
        file_upload = gr.File(label=i18n("Загрузите zip-файл содержащий в себе pth и index"))
        import_button = gr.Button(i18n("Импортировать"))
        import_status = gr.Textbox(label=i18n("Консоль"))
        import_button.click(fn=import_button_click, inputs=file_upload, outputs=import_status)