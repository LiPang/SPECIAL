import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from segearth_segmentor import SegEarthSegmentation
from scipy.io import *

def rsshow(I, scale=0.005):
    low, high = np.quantile(I, [scale, 1-scale])
    I[I > high] = high
    I[I < low] = low
    I = (I-low)/(high-low)
    return I

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
def generate_classification_report(predicted_labels, true_labels):
    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    overall_precision = precision_score(true_labels, predicted_labels, average='weighted')
    overall_recall = recall_score(true_labels, predicted_labels, average='weighted')
    overall_f1 = f1_score(true_labels, predicted_labels, average='weighted')

    class_report = classification_report(true_labels, predicted_labels, output_dict=True)

    report_df = pd.DataFrame(class_report).transpose()

    # 添加总体指标
    summary_data = {
        'overall_accuracy': overall_accuracy,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1
    }
    summary_df = pd.DataFrame([summary_data], index=['Summary'])

    # 合并详细和总体指标
    overall_df = pd.concat([report_df, summary_df])

    cm = confusion_matrix(true_labels, predicted_labels)

    return overall_df
    # # 保存为 CSV 文件
    # output_file = "classification_metrics.csv"
    # overall_df.to_csv(output_file, index=True)
    #
    # print(f"分类指标已保存到 {output_file}")


# img = Image.open('/data01/pl/HSITask/data/HSI/data_256_256_rgb/h2sr_00692.png')
# # img = Image.open('/data01/pl/HSITask/data/Mydata/hengyang/1000/002368_(12531274, 3116197, 12531544, 3116467, 1000).tif')
img = Image.open('/home/pl/HSITask/UnsupervisedHSIClassification/Baselines/SegEarth-OV/demo/oem_koeln_50.tif')
# img = img.convert("RGB").resize((224, 224))
img = np.array(img)

# img = loadmat("/data01/pl/HSITask/data/Urban/data_mat.mat")['data']
# img = rsshow(img[..., [20, 12, 4]], 0.001)
# label = loadmat("/data01/pl/HSITask/data/Urban/label_mat.mat")['data']
# mapping = loadmat("/data01/pl/HSITask/data/Urban/class_mapping.mat")['data'][0]
# label = mapping[label]

# name_list = ['background', 'soil,bareland,barren', 'grass', 'road',
#              'tree,forest', 'water,river', 'cropland', 'building,roof,house']
name_list = ["farmland", "soil", "road", "buildings", "water,river,lake", "vegetation"]

# with open('./configs/my_name.txt', 'w') as writers:
#     for i in range(len(name_list)):
#         if i == len(name_list)-1:
#             writers.write(name_list[i])
#         else:
#             writers.write(name_list[i] + '\n')
# writers.close()


img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    # transforms.Resize((256, 256))
])(img)

img_tensor = img_tensor.unsqueeze(0).to('cuda')

model = SegEarthSegmentation(
    clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
    vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
    model_type='SegEarth',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
    ignore_residual=True,
    feature_up=True,
    feature_up_cfg=dict(
        model_name='jbu_one',
        model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
    cls_token_lambda=-0.3,
    name_path=name_list,
    prob_thd=0.1,
)

seg_pred = model.predict(img_tensor, data_samples=None)
seg_pred = seg_pred.data.cpu().numpy().squeeze(0)
# seg_pred = seg_pred + 1
# label_vec, pred_label_vec = label.flatten(), seg_pred.flatten()
# report = generate_classification_report(pred_label_vec[label_vec > 0],
#                                         label_vec[label_vec > 0])
#
# from matplotlib import cm
# from matplotlib.patches import Patch
# label = seg_pred
# unique_values = np.unique(label).astype(np.int32)
# colormap = cm.get_cmap('Set3', 10)
# colors = {value: colormap(value)[:3] + (0.4,) for value in unique_values}
# overlay = np.zeros((label.shape[0], label.shape[1], 4))
# for value, color in colors.items():
#     overlay[label == value] = color
# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# axes[0].imshow(img)
# axes[0].axis('off')
# axes[1].imshow(overlay)
# axes[1].axis('off')
# legend_elements = [Patch(facecolor=color[:3], alpha=color[3], label=f'{name_list[int(value) - 1]}')
#                    for value, color in colors.items() if value != 0]
# axes[1].legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.05, 0.5))
# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].axis('off')
ax[1].imshow(seg_pred, cmap='viridis')
ax[1].axis('off')
# plt.tight_layout()
# plt.show()
plt.savefig('seg_pred.png', bbox_inches='tight')