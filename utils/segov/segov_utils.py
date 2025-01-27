from torchvision import transforms
from .segearth_segmentor import SegEarthSegmentation


def build_segov(name_list):
    model = SegEarthSegmentation(
        clip_type='CLIP',  # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
        vit_type='ViT-B/16',  # 'ViT-B/16', 'ViT-L-14'
        model_type='SegEarth',  # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
        ignore_residual=True,
        feature_up=True,
        feature_up_cfg=dict(
            model_name='jbu_one',
            model_path='utils/segov/simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
        cls_token_lambda=-0.3, # -0.3
        name_path=name_list,
        prob_thd=0.1,
    )
    return model

def segov_process(img, model, slide=True, **kwargs):
    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])(img)

    img_tensor = img_tensor.unsqueeze(0).to('cuda')
    seg_pred, seg_logits = model.predict(img_tensor, data_samples=None, slide=slide, **kwargs)
    return seg_pred, seg_logits