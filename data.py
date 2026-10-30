import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import random
import os
import utils
import features
import project_paths as pp

class ImageFeatDataset(Dataset):
    """
    Provide (image, label) pair for association matrix optimization,
    where image is a PIL Image
    """
    def __init__(self, img_feat, label, on_gpu):
        device = "cuda" if (on_gpu and torch.cuda.is_available()) else "cpu"
        self.img_feat = img_feat.to(device)
        self.labels = label.to(device)

    def __len__(self):
        return len(self.img_feat)

    def __getitem__(self, index):
        img_feat_anchor, label = self.img_feat[index], self.labels[index]

        return img_feat_anchor, label 

class DataModule(pl.LightningDataModule):
    """
    It prepares image and concept CLIP features given config of one dataset.
    """
    def __init__(
            self,
            num_concept,
            data_root,
            clip_model,
            img_split_path,
            img_root,
            n_shots,
            concept_raw_path, 
            concept2cls_path, 
            concept_select_fn, 
            cls_names_path,
            batch_size,
            use_txt_norm=False,
            use_img_norm=False,
            num_workers=0,
            img_ext='.jpg',
            clip_ckpt=None,
            on_gpu=False,
            force_compute=True,
            use_cls_name_init='none',
            use_cls_sim_prior='none',
            remove_cls_name=False,
            pretrained_model=None,
            concept2type=False,
            pearson_weight=None,
            feature_dir=None,
            strict=False):
        super().__init__()
        
        self.pretrained_model = pretrained_model
        self.is_concept2type = concept2type
        self.pearson_weight=pearson_weight
        
        # image feature is costly to compute, so it will always be cached
        self.force_compute = force_compute 
        self.use_txt_norm = use_txt_norm 
        self.use_img_norm = use_img_norm
        self.use_cls_name_init = use_cls_name_init
        self.use_cls_sim_prior = use_cls_sim_prior
        self.remove_cls_name = remove_cls_name
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True, parents=True)
        # strict=True (test/eval): never regenerate caches, fail if they are stale
        self.strict = strict
        self.feature_dir = Path(feature_dir) if feature_dir else Path(img_split_path) / 'features'
        self.img_split_path = Path(img_split_path)
        self.img_split_path.mkdir(exist_ok=True, parents=True)

        # all variables save_dir that will be created inside this module
        if self.use_cls_name_init != 'none':
            self.init_weight_save_dir = self.data_root.joinpath('init_weight.pth')
        if self.use_cls_sim_prior != 'none':
            self.cls_sim_save_dir = self.data_root.joinpath('cls_sim.pth')
        self.select_idx_save_dir = self.data_root.joinpath(
            'select_idx.pth')  # selected concept indices
        self.concept_pair_save_dir = self.data_root.joinpath(
            'concept_pair.pth')  # concept pairs for selected concepts
        
        self.concepts_raw_save_dir = self.data_root.joinpath(
            'concepts_raw_selected.npy')
        self.concept2cls_save_dir = self.data_root.joinpath(
            'concept2cls_selected.npy')
        self.concept_feat_save_dir = self.data_root.joinpath(
            'concepts_feat_{}.pth'.format(clip_model.replace('/','-')))

        self.clip_model = clip_model
        self.clip_ckpt = clip_ckpt
        self.cls_names = np.load(cls_names_path).tolist() # for reference, the mapping between indices and names
        self.num_concept = num_concept


        # handling image related data
        self.splits = {
            split: utils.pickle_load(
                self.img_split_path.joinpath(
                    'class2images_{}.p'.format(split)))
            for split in ['train', 'val', 'test']
        }

        self.n_shots = n_shots
        self.img_root = Path(img_root)
        self.img_ext = img_ext
        self.prepare_img_feat(self.splits, self.n_shots, self.clip_model, self.clip_ckpt)

        if self.n_shots != "all": 
            self.num_images_per_class = [self.n_shots] * len(self.splits['train'])
        else:
            self.num_images_per_class = [len(images) for _, images in self.splits['train'].items()]

        # handling concept related data
        self.concepts_raw = np.load(concept_raw_path)
        self.concept2cls = np.load(concept2cls_path)
        self.concept2cls_path = concept2cls_path
        
        # TODO: remove duplication
        self.concepts_raw, idx = self.preprocess(self.concepts_raw, self.cls_names)
        self.concept2cls = self.concept2cls[idx]
        self.concept2type = None
        if self.is_concept2type:
            self.concept2type = np.load(self.concept2cls_path.replace('concept2cls.npy', 'concept2type.npy'))
            self.concept2type = self.concept2type[idx] 
        
        self.concept_select_fn = concept_select_fn

        if self.n_shots != "all":
            assert len(self.img_feat['train']) == len(self.cls_names) * self.n_shots

        self.prepare_txt_feat(self.concepts_raw, self.clip_model, self.clip_ckpt)

        # TODO: manually comment the negative value issue in apricot
        # https://github.com/YueYANG1996/LaBo/issues/1#issuecomment-1583107414
        self.select_concept(self.concept_select_fn, self.img_feat['train'], self.concept_feat, self.n_shots, 
                self.num_concept, self.concept2cls, self.clip_ckpt, self.num_images_per_class)

        # save all raw concepts and corresponding classes as a reference
        np.save(self.concepts_raw_save_dir, self.concepts_raw) # concept as text
        np.save(self.concept2cls_save_dir, self.concept2cls) # concept for target class

        if self.use_cls_name_init != 'none':
            self.gen_init_weight_from_cls_name(self.cls_names, self.concepts_raw[self.select_idx])

        if self.use_cls_sim_prior != 'none':
            split = 'train'
            self.gen_mask_from_img_sim(self.img_feat[split], self.n_shots, self.label[split][::self.n_shots])

        # parameters for dataloader
        self.bs = batch_size
        self.num_workers = num_workers
        self.on_gpu = on_gpu

    def check_pattern(self, concepts, pattern):
        """
        Return a boolean array where it is true if one concept contains the pattern 
        """
        return np.char.find(concepts, pattern) != -1

    def check_no_cls_names(self, concepts, cls_names):
        res = np.ones(len(concepts), dtype=bool)
        for cls_name in cls_names: 
            no_cls_name = ~self.check_pattern(concepts, cls_name)
            res = res & no_cls_name 
        return res

    def preprocess(self, concepts, cls_names=None):
        """
        concepts: numpy array of strings of concepts
        
        This function checks all input concepts, remove duplication, and 
        remove class names if necessary
        """
        concepts, left_idx = np.unique(concepts, return_index=True)
        if self.remove_cls_name: 
            print('remove cls name')
            is_good = self.check_no_cls_names(concepts, cls_names)
            concepts = concepts[is_good]
            left_idx = left_idx[is_good]
        return concepts, left_idx

    def gen_init_weight_from_cls_name(self, cls_names, concepts):
        # always use unnormalized text feature for more accurate class-concept assocation
        num_cls = len(cls_names)
        num_concept_per_cls = self.num_concept // num_cls
        cls_name_feat = utils.prepare_txt_feat(cls_names, clip_model_name=self.clip_model, ckpt_path=self.clip_ckpt)
        concept_feat = utils.prepare_txt_feat(concepts, clip_model_name=self.clip_model, ckpt_path=self.clip_ckpt)
        dis = torch.cdist(cls_name_feat, concept_feat)
        # select top k concept with smallest distanct to the class name
        _, idx = torch.topk(dis, num_concept_per_cls, largest=False)
        init_weight = torch.zeros((num_cls, self.num_concept))
        init_weight.scatter_(1, idx, 1)
        torch.save(init_weight, self.init_weight_save_dir)

    def gen_mask_from_img_sim(self, img_feat, n_shots, label):
        print('generate cls sim mask')
        num_cls = len(img_feat) // n_shots
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-7)
        img_sim = img_feat @ img_feat.T
        class_sim = torch.empty((num_cls, num_cls))
        for i, row_split in enumerate(torch.split(img_sim, n_shots, dim=0)):
            for j, col_split in enumerate(torch.split(row_split, n_shots, dim=1)):
                class_sim[label[i], label[j]] = torch.mean(col_split)

        good = class_sim >= torch.quantile(class_sim, 0.95, dim=-1)
        final_sim = torch.zeros(class_sim.shape)
        for i in range(num_cls):
            for j in range(num_cls):
                if i == j: final_sim[i, j] = 1
                elif good[i, j] == True: final_sim[i, j] = class_sim[i, j]

        torch.save(final_sim, self.cls_sim_save_dir)
        self.class_sim = final_sim

    def select_concept(self, concept_select_fn, img_feat_train, concept_feat, n_shots, num_concepts, concept2cls, clip_ckpt, num_images_per_class):
        """Concept selection uses TRAIN features only. The result is cached with a
        fingerprint of everything it depends on; a mismatch recomputes (or raises in strict mode)."""
        meta = {"kind": "select_idx",
                "train_fingerprint": self.split_fingerprint['train'],
                "concept_hash": features.concept_hash(self.concepts_raw),
                "clip_sha256": features.expected_clip_sha(self.clip_model),
                "num_concepts": int(num_concepts),
                "pearson_weight": self.pearson_weight,
                "select_fn": getattr(concept_select_fn, '__name__', str(concept_select_fn)),
                "use_img_norm": bool(self.use_img_norm), "use_txt_norm": bool(self.use_txt_norm)}
        cached = None
        try:
            cached = features.load_checked(self.select_idx_save_dir, meta)
        except features.StaleCacheError:
            if self.strict:
                raise
        if cached is not None:
            self.select_idx = cached
            return
        if self.strict:
            raise features.StaleCacheError(f"{self.select_idx_save_dir} missing in strict mode")
        print('select concept')
        self.select_idx, selected_concept_pairs = concept_select_fn(
            img_feat_train, concept_feat, concept2cls, num_concepts, num_images_per_class,
            concept2type=self.concept2type, pearson_weight=self.pearson_weight)
        features.save_checked(self.select_idx, self.select_idx_save_dir, meta)
        if selected_concept_pairs is not None:
            torch.save(torch.tensor(selected_concept_pairs), self.concept_pair_save_dir)
        elif self.concept_pair_save_dir.exists():
            self.concept_pair_save_dir.unlink()  # stale pairs would be read by the model
        print('concepts are saved in', self.select_idx_save_dir)

    def _txt_meta(self, concepts_raw):
        return {"kind": "concept_feats", "clip_model": self.clip_model,
                "clip_sha256": features.expected_clip_sha(self.clip_model),
                "concept_hash": features.concept_hash(concepts_raw),
                "n": int(len(concepts_raw))}

    def prepare_txt_feat(self, concepts_raw, clip_model, clip_ckpt):
        """Concept text features, cached together with a hash of the concept list."""
        meta = self._txt_meta(concepts_raw)
        cached = features.load_checked(self.concept_feat_save_dir, meta)  # raises if stale
        if cached is None:
            if self.strict:
                raise features.StaleCacheError(f"{self.concept_feat_save_dir} missing in strict mode")
            print('prepare txt feat')
            cached = utils.prepare_txt_feat(concepts_raw, clip_model_name=clip_model, ckpt_path=None)
            features.save_checked(cached, self.concept_feat_save_dir, meta)
        self.concept_feat = cached.clone()
        if self.use_txt_norm:
            self.concept_feat /= self.concept_feat.norm(dim=-1, keepdim=True)

    def prepare_img_feat(self, splits, n_shots, clip_model, clip_ckpt):
        """Train / val / test features are cached separately and verified against the
        current split files and CLIP weights (see features.py)."""
        assert n_shots == 'all', "cached feature pipeline supports n_shots='all' only"
        self.img_feat, self.label, self.img_names = {}, {}, {}
        self.split_fingerprint = {}
        for mode in ['train', 'val', 'test']:
            feats, labels, names = features.get_split_features(
                mode, splits[mode], self.cls_names, self.feature_dir, clip_model,
                img_root=str(self.img_root) if str(self.img_root) not in ('', '.') else None,
                allow_extract=not self.strict)
            feats = feats.clone()
            if self.use_img_norm:
                feats /= feats.norm(dim=-1, keepdim=True)
            self.img_feat[mode], self.label[mode], self.img_names[mode] = feats, labels, names
            self.split_fingerprint[mode] = features.split_fingerprint(names, labels.tolist())

    def setup(self, stage):
        """
        Set up datasets for dataloader to load from. Depending on the need, return either:
        - (img_feat, label), concept_feat will be loaded in the model
        - (the dot product between img_feat and concept_feat, label)
        - if allowing grad to image, provide (image, label)
        - if allowing grad to text, compute concept_feat inside the model        
        """
        b = self.concepts_raw[self.select_idx[:self.num_concept]]
        concept_label = self.concept2cls[self.select_idx[:self.num_concept]]
        selected_concepts = {cl : [] for cl in concept_label}
        i = 0
        for cl in concept_label:
            selected_concepts[cl].append(str(b[i]))
            i+=1

        with open(self.data_root.joinpath('selected_concepts.json'), 'w') as f:
            f.write(json.dumps(selected_concepts, indent=4))

        self.datasets = {
            mode: ImageFeatDataset(self.img_feat[mode], 
                                    self.label[mode],
                                    self.on_gpu)
            for mode in ['train', 'val', 'test']
        }

    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'],
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False)

    def val_dataloader(self):
        return DataLoader(
            self.datasets['val'],
            batch_size=self.bs,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False)

    def test_dataloader(self):
        return DataLoader(
            self.datasets['test'],
            batch_size=self.bs,
            num_workers=self.num_workers if not self.on_gpu else 0,
            pin_memory=True if not self.on_gpu else False)

    def predict_dataloader(self):
        return self.test_dataloader()







