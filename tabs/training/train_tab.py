import gradio as gr
from .train import *

def train_tab():
    with gr.Row():
        exp_dir1 = gr.Textbox(
            label=i18n("Enter the experiment name"), value="mi-test"
        )
        author = gr.Textbox(label=i18n("Model Author (Nullable)"))
        np7 = gr.Slider(
            minimum=0,
            maximum=config.n_cpu,
            step=1,
            label=i18n(
                "Number of CPU processes used for pitch extraction and data processing"
            ),
            value=int(np.ceil(config.n_cpu / 1.5)),
            interactive=True,
        )
    with gr.Row():
        sr2 = gr.Radio(
            label=i18n("Target sample rate"),
            choices=["40k", "48k"],
            value="40k",
            interactive=True,
        )
        if_f0_3 = gr.Radio(
            label=i18n(
                "Whether the model has pitch guidance (required for singing, optional for speech)"
            ),
            choices=[i18n("Yes"), i18n("No")],
            value=i18n("Yes"),
            interactive=True,
        )
        version19 = gr.Radio(
            label=i18n("Version"),
            choices=["v1", "v2"],
            value="v2",
            interactive=True,
            visible=True,
        )
    gr.Markdown(
        value=i18n(
            "### Step 2. Audio processing. \n#### 1. Slicing.\nAutomatically traverse all files in the training folder that can be decoded into audio and perform slice normalization. Generates 2 wav folders in the experiment directory. Currently, only single-singer/speaker training is supported."
        )
    )
    with gr.Row():
        with gr.Column():
            trainset_dir4 = gr.Textbox(
                label=i18n("Enter the path of the training folder"),
            )
            spk_id5 = gr.Slider(
                minimum=0,
                maximum=4,
                step=1,
                label=i18n("Please specify the speaker/singer ID"),
                value=0,
                interactive=True,
            )
            but1 = gr.Button(i18n("Process data"), variant="primary")
        with gr.Column():
            info1 = gr.Textbox(label=i18n("Output information"), value="")
            but1.click(
                preprocess_dataset,
                [trainset_dir4, exp_dir1, sr2, np7],
                [info1],
                api_name="train_preprocess",
            )
    gr.Markdown(
        value=i18n(
            "#### 2. Feature extraction.\nUse CPU to extract pitch (if the model has pitch), use GPU to extract features (select GPU index)."
        )
    )
    with gr.Row():
        with gr.Column():
            gpu_info9 = gr.Textbox(
                label=i18n("GPU Information"),
                value=gpu_info,
                visible=F0GPUVisible,
            )
            gpus6 = gr.Textbox(
                label=i18n(
                    "Enter the GPU index(es) separated by '-', e.g., 0-1-2 to use GPU 0, 1, and 2"
                ),
                value=gpus,
                interactive=True,
                visible=F0GPUVisible,
            )
            gpus_rmvpe = gr.Textbox(
                label=i18n(
                    "Enter the GPU index(es) separated by '-', e.g., 0-0-1 to use 2 processes in GPU0 and 1 process in GPU1"
                ),
                value="%s-%s" % (gpus, gpus),
                interactive=True,
                visible=F0GPUVisible,
            )
            f0method8 = gr.Radio(
                label=i18n(
                    "Select the pitch extraction algorithm: when extracting singing, you can use 'pm' to speed up. For high-quality speech with fast performance, but worse CPU usage, you can use 'dio'. 'harvest' results in better quality but is slower.  'rmvpe' has the best results and consumes less CPU/GPU"
                ),
                choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                value="rmvpe_gpu",
                interactive=True,
            )
        with gr.Column():
            but2 = gr.Button(i18n("Feature extraction"), variant="primary")
            info2 = gr.Textbox(label=i18n("Output information"), value="")
        f0method8.change(
            fn=change_f0_method,
            inputs=[f0method8],
            outputs=[gpus_rmvpe],
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
            [info2],
            api_name="train_extract_f0_feature",
        )
    gr.Markdown(
        value=i18n(
            "### Step 3. Start training.\nFill in the training settings and start training the model and index."
        )
    )
    with gr.Row():
        with gr.Column():
            save_epoch10 = gr.Slider(
                minimum=1,
                maximum=50,
                step=1,
                label=i18n("Save frequency (save_every_epoch)"),
                value=5,
                interactive=True,
            )
            total_epoch11 = gr.Slider(
                minimum=2,
                maximum=1000,
                step=1,
                label=i18n("Total training epochs (total_epoch)"),
                value=20,
                interactive=True,
            )
            batch_size12 = gr.Slider(
                minimum=1,
                maximum=40,
                step=1,
                label=i18n("Batch size per GPU"),
                value=default_batch_size,
                interactive=True,
            )
            if_save_latest13 = gr.Radio(
                label=i18n(
                    "Save only the latest '.ckpt' file to save disk space"
                ),
                choices=[i18n("Yes"), i18n("No")],
                value=i18n("No"),
                interactive=True,
            )
            if_cache_gpu17 = gr.Radio(
                label=i18n(
                    "Cache all training sets to GPU memory. Caching small datasets (less than 10 minutes) can speed up training, but caching large datasets will consume a lot of GPU memory and may not provide much speed improvement"
                ),
                choices=[i18n("Yes"), i18n("No")],
                value=i18n("No"),
                interactive=True,
            )
            if_save_every_weights18 = gr.Radio(
                label=i18n(
                    "Save a small final model to the 'weights' folder at each save point"
                ),
                choices=[i18n("Yes"), i18n("No")],
                value=i18n("No"),
                interactive=True,
            )
        with gr.Column():
            pretrained_G14 = gr.Textbox(
                label=i18n("Load pre-trained base model G path"),
                value="assets/pretrained_v2/f0G40k.pth",
                interactive=True,
            )
            pretrained_D15 = gr.Textbox(
                label=i18n("Load pre-trained base model D path"),
                value="assets/pretrained_v2/f0D40k.pth",
                interactive=True,
            )
            gpus16 = gr.Textbox(
                label=i18n(
                    "Enter the GPU index(es) separated by '-', e.g., 0-1-2 to use GPU 0, 1, and 2"
                ),
                value=gpus,
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

            but3 = gr.Button(i18n("Train model"), variant="primary")
            but4 = gr.Button(i18n("Train feature index"), variant="primary")
            but5 = gr.Button(i18n("One-click training"), variant="primary")
    with gr.Row():
        info3 = gr.Textbox(label=i18n("Output information"), value="")
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
                author,
            ],
            info3,
            api_name="train_start",
        )
        but4.click(train_index, [exp_dir1, version19], info3)
        but5.click(
            train1key,
            [
                exp_dir1,
                sr2,
                if_f0_3,
                trainset_dir4,
                spk_id5,
                np7,
                f0method8,
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
                gpus_rmvpe,
                author,
            ],
            info3,
            api_name="train_start_all",
        )