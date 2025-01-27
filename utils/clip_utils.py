from torchvision import transforms

def _convert_to_rgb(image):
    return image.convert('RGB')

def get_preprocess(image_resolution=224):
    normalize = transforms.Normalize(
        mean=[0.406, 0.423, 0.390], std=[0.188, 0.175, 0.185]
    )
    preprocess_val = transforms.Compose([
        transforms.Resize(
            size=image_resolution,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(image_resolution),
        _convert_to_rgb,
        transforms.ToTensor(),
        normalize,
    ])
    return preprocess_val


def build_georsclip(device, image_resolution=224):
    import open_clip
    ckpt_path = "/data01/pl/HSITask/checkpoints/GeoRSCLIP/RS5M_ViT-L-14.pt"
    img_preprocess = get_preprocess(
        image_resolution=image_resolution,
    )
    model, _, _ = open_clip.create_model_and_transforms("ViT-L/14",
                                                        pretrained=ckpt_path)
    model.to(device)
    return img_preprocess, model