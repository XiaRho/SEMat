import gradio as gr
from gradio_image_prompter import ImagePrompter
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import cv2
import numpy as np
import torch
import os

model_choice = {
    'SAM': '../2025_ICLR_SEMat/checkpoints/SAM_240911-2030_247_model_final.pth', 
    'HQ-SAM': '../2025_ICLR_SEMat/checkpoints/HQ-SAM_240907-1324_222_model_final.pth', 
    'SAM2': '../2025_ICLR_SEMat/checkpoints/SAM2_240912-1743_251_model_0053999.pth'
}

def load_model(model_type='HQ-SAM'):
    assert model_type in model_choice.keys()
    config_path = './configs/SEMat_{}.py'.format(model_type)
    cfg = LazyConfig.load(config_path)

    if hasattr(cfg.model.sam_model, 'ckpt_path'):
        cfg.model.sam_model.ckpt_path = None
    else:
        cfg.model.sam_model.checkpoint = None
    model = instantiate(cfg.model)
    if model.lora_rank is not None:
        model.init_lora()
    model.to(cfg.train.device)
    DetectionCheckpointer(model).load(model_choice[model_type])
    model.eval()
    return model, model_type

def transform_image_bbox(prompts):
    if len(prompts["points"]) != 1:
        raise gr.Error("Please input only one BBox.", duration=5)
    [[x1, y1, idx_3, x2, y2, idx_6]] = prompts["points"]
    if idx_3 != 2 or idx_6 != 3:
        raise gr.Error("Please input BBox instead of point.", duration=5)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    img = prompts["image"]
    ori_H, ori_W, _ = img.shape

    scale = 1024 * 1.0 / max(ori_H, ori_W)
    new_H, new_W = ori_H * scale, ori_W * scale
    new_W = int(new_W + 0.5)
    new_H = int(new_H + 0.5)

    img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    padding = np.zeros([1024, 1024, 3], dtype=img.dtype)
    padding[: new_H, : new_W, :] = img
    img = padding
    # img = img[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.0
    img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0

    [[x1, y1, _, x2, y2, _]] = prompts["points"]
    x1, y1, x2, y2 = int(x1 * scale + 0.5), int(y1 * scale + 0.5), int(x2 * scale + 0.5), int(y2 * scale + 0.5)
    bbox = np.clip(np.array([[x1, y1, x2, y2]]) * 1.0, 0, 1023.0)

    return img, bbox, (ori_H, ori_W), (new_H, new_W)

if __name__ == '__main__':

    model, model_type = load_model()

    def inference_image(prompts, input_model_type):

        global model_type
        global model

        if input_model_type != model_type:
            gr.Info('Loading SEMat of {} version.'.format(input_model_type), duration=5)
            _model, _ = load_model(input_model_type)
            model_type = input_model_type
            model = _model

        image, bbox, ori_H_W, pad_H_W = transform_image_bbox(prompts)
        input_data = {
            'image': torch.from_numpy(image)[None].to(model.device),
            'bbox': torch.from_numpy(bbox)[None].to(model.device),
        }

        with torch.no_grad():
            inputs = model.preprocess_inputs(input_data) 
            images, bbox, gt_alpha, trimap, condition = inputs['images'], inputs['bbox'], inputs['alpha'], inputs['trimap'], inputs['condition']

            if model.backbone_condition:
                condition_proj = model.condition_embedding(condition) 
            elif model.backbone_bbox_prompt is not None or model.bbox_prompt_all_block is not None:
                condition_proj = bbox
            else:
                condition_proj = None

            low_res_masks, pred_alphas, pred_trimap, sam_hq_matting_token = model.forward_samhq_and_matting_decoder(images, bbox, condition_proj)


        output_alpha = np.uint8(pred_alphas[0, 0][:pad_H_W[0], :pad_H_W[1], None].repeat(1, 1, 3).cpu().numpy() * 255)

        return output_alpha

    with gr.Blocks() as demo:

        with gr.Row():
            with gr.Column(scale=45):
                img_in = ImagePrompter(type='numpy', show_label=False, label="Input Image")
                
            with gr.Column(scale=45):
                img_out = gr.Image(type='pil', label="Pred. Alpha")

        with gr.Row():
            with gr.Column(scale=45):
                input_model_type = gr.Dropdown(list(model_choice.keys()), value='HQ-SAM', label="Trained SEMat Version")

            with gr.Column(scale=45):
                bt = gr.Button()

        bt.click(inference_image, inputs=[img_in, input_model_type], outputs=[img_out]) 

        # example_files = os.listdir('./demo_imgs')
        # example_files.sort()
        # # example_files = [{'image': cv2.imread(os.path.join('./demo_imgs', filename)), 'points': None}  for filename in example_files]
        # examples = gr.Examples(examples=example_files, inputs=[img_in])

demo.launch()
