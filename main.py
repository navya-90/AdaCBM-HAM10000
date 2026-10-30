import utils
import pytorch_lightning as pl
import torch
import numpy as np
import json
import numpy as np
import os
from datetime import datetime
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import project_paths

device = "cuda" if torch.cuda.is_available() else "cpu"

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

def save_npy_files(class2concepts, save_dir):
    # sort class name to make sure they are in the same order, to avoid potential problem
    class_names = sorted(list(class2concepts.keys()))
    num_concept = sum([len(concepts) for concepts in class2concepts.values()])
    concept2cls = np.zeros(num_concept)
    i = 0
    all_concepts = []
    for class_name, concepts in class2concepts.items():
        class_idx = class_names.index(class_name)
        for concept in concepts:
            all_concepts.append(concept)
            concept2cls[i] = class_idx
            i += 1
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'concepts_raw.npy'), np.array(all_concepts))
    np.save(os.path.join(save_dir, 'cls_names.npy'), np.array(class_names))
    np.save(os.path.join(save_dir, 'concept2cls.npy'), concept2cls)

def our_main(cfg):
    start_time = datetime.now()
    from models.asso_opt.asso_our import OurConcept
    from models.select_concept.select_algo import our_selection
    from data import DataModule
    import random
    
    MODEL = OurConcept

    cfg.concept2type = False
    cfg.use_txt_norm = False
    cfg.use_img_norm = False
    cfg.use_residual = False
    
    if type(cfg.seed)==list:
        cfg.seed = int(cfg.seed[0])
    if type(cfg.concept_select_fn)==list:
        cfg.concept_select_fn = cfg.concept_select_fn[0]
    if type(cfg.num_concept)==list:
        cfg.num_concept = int(cfg.num_concept[0])
    
    print('Configurations:', cfg)
    print('Model:', MODEL.__name__)

    class2concepts = json.load(open(cfg.concept_path, "r"))

    save_npy_files(class2concepts, cfg.concept_root)
    
    if not cfg.seed: cfg.seed = 123
    pl.seed_everything(int(cfg.seed))
    
    concept_select_fn = our_selection

    data_module = DataModule(
        cfg.num_concept,
        cfg.data_root,
        cfg.clip_model,
        cfg.img_split_path,
        cfg.img_path,
        cfg.n_shots,
        cfg.raw_sen_path,
        cfg.concept2cls_path,
        concept_select_fn,
        cfg.cls_name_path,
        cfg.bs,
        on_gpu=cfg.on_gpu,
        num_workers=cfg.num_workers if 'num_workers' in cfg else 0,
        img_ext=cfg.img_ext if 'img_ext' in cfg else '.jpg',
        clip_ckpt=None,  # cfg.ckpt_path is the *Lightning* checkpoint (test mode), never CLIP weights
        use_txt_norm=cfg.use_txt_norm if 'use_txt_norm' in cfg else False, 
        use_img_norm=cfg.use_img_norm if 'use_img_norm' in cfg else False,
        use_cls_name_init=cfg.cls_name_init if 'cls_name_init' in cfg else 'none',
        use_cls_sim_prior=cfg.cls_sim_prior if 'cls_sim_prior' in cfg else 'none',
        remove_cls_name=cfg.remove_cls_name if 'remove_cls_name' in cfg else True,
        concept2type=cfg.concept2type,
        pearson_weight=cfg.pearson_weight if 'pearson_weight' in cfg else None,
        feature_dir=cfg.feature_dir,
        strict=bool(cfg.test)
    )

    model = MODEL(cfg, init_weight=torch.load(cfg.init_weight_path) if 'init_weight_path' in cfg else None)

    check_interval = 1
    
    if cfg.test:
        ckpt_path = cfg.ckpt_path
        print('load ckpt: {}'.format(ckpt_path))
        model = MODEL.load_from_checkpoint(str(ckpt_path), map_location=device, weights_only=False, cfg=cfg)
        trainer = pl.Trainer(accelerator="auto")
        trainer.test(model, data_module)
        # test_acc = round(100 * float(model.total_test_acc), 2)
        return

    print("check interval = {}".format(check_interval))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.work_dir,
        filename='{epoch}-{step}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        save_last=True,
        every_n_epochs=check_interval)
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    
    tb_logger = TensorBoardLogger(save_dir=cfg.log_dir)
    trainer = pl.Trainer(max_epochs=int(cfg.max_epochs) if 'max_epochs' in cfg else 1000,
        callbacks=[checkpoint_callback, lr_monitor], 
        check_val_every_n_epoch=check_interval, 
        default_root_dir=cfg.work_dir,
        accelerator="auto",
        logger=tb_logger)
        
    trainer.fit(model, data_module)

    delta = datetime.now() - start_time
    # time difference in seconds
    print(f"Training time is {delta.total_seconds()/60} minutes")
    
    # print('Best model path:', checkpoint_callback.best_model_path)
    
    print('Evaluated on Last model:')
    result = trainer.test(model, datamodule=data_module)
    
    # last.ckpt (final epoch) and the best-val checkpoint stay in cfg.work_dir; test accuracy is
    # deliberately NOT put in file names so it cannot drive model selection.
    
    with open(os.path.join(cfg.metrics_dir, 'baseline_test_last.json'), "w") as f:
        json.dump({'run': cfg.run_name, 'seed': int(cfg.seed), 'checkpoint': 'last',
                   'val_acc_ckpt_monitor': 'val_acc (validation split)',
                   'test': {k: float(v) for k, v in result[0].items()}}, f, indent=2)
    
    # print('best model')
    # trainer.test(datamodule=data_module)
    

if __name__ == "__main__":
    try:
        from mmcv import DictAction, Config
    except ImportError:
        import compat
        from compat import DictAction, Config

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./cfg/HAM10000/HAM10000_allshot_fac.py', help='path to config file')
    parser.add_argument('--work-dir', default=None,
                        help='override checkpoint directory (default: <results>/runs/<run-name>/checkpoints)')
    parser.add_argument('--run-name', default='adacbm_ham10000', help='name of this run under <results>/runs/')
    parser.add_argument('--results-dir', default=None, help='results root (default: $RESULTS_DIR or ./results)')
    parser.add_argument('--func', default='our_main', help='which task to run')
    parser.add_argument('--test',
                        action='store_true',
                        help='whether to enable test mode')
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        help='overwrite parameters in cfg from commandline')
    args = parser.parse_args()
    cfg = Config.fromfile(args.cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.test = args.test
    cfg = project_paths.apply_to_cfg(cfg, run_name=args.run_name, results_dir=args.results_dir,
                                     work_dir=args.work_dir)
    if not args.test:
        cfg.dump(os.path.join(cfg.run_dir, 'config_snapshot.py'))
        
    main = eval(args.func)
    main(cfg)