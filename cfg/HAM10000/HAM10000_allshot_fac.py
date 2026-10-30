# NOTE: every filesystem path here is overwritten by project_paths.apply_to_cfg()
# (results dir, feature dir, image root, split dir, concept dir). Do not hardcode paths.
proj_name = "HAM10000"
img_ext = ''
num_cls = 7

## data loader
on_gpu = True

# concept select
num_concept = num_cls * 10   # 70 concepts (10 per class)

# weight matrix fitting
lr = 0.0005
max_epochs = 300

# CLIP Backbone
clip_model = 'ViT-L/14'
n_shots = "all"
cls_names = ['actinic keratoses', 'basal cell carcinoma', 'benign keratosis-like lesions',
            'dermatofibroma', 'melanocytic nevi',
            'melanoma', 'vascular lesions']
dataset = 'HAM10000'
bs = 256
num_layers = 2
freeze_backbone = True
use_img_norm = False
use_txt_norm = False
use_rand_init = False
init_val = 1.0

seed = 123
pearson_weight = 0.9
concept_path = 'gpt_concepts/HAM10000.json'   # relative to the repo root
