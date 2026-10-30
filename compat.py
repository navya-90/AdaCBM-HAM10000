"""
Compatibility module providing pure-Python replacements for mmcv dependencies.
Enables AdaCBM to run seamlessly on Windows, Linux, and MacOS without C++ compilation requirements.
"""

import os
import sys
import types
import runpy
import ast
import argparse


class ConfigDict(dict):
    """A dictionary subclass that allows attribute-style access and unpickling compatibility."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, ConfigDict):
                self[k] = ConfigDict(v)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.update(state)
        elif isinstance(state, tuple):
            for item in state:
                if isinstance(item, dict):
                    self.update(item)


class Config(ConfigDict):
    """Configuration loader matching mmcv.Config interface."""
    @classmethod
    def fromfile(cls, filename):
        cfg_dict = runpy.run_path(str(filename))
        clean_dict = {k: v for k, v in cfg_dict.items() if not k.startswith('__')}
        cfg = cls(clean_dict)
        cfg.filename = str(filename)
        return cfg

    def dump(self, filename):
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, 'w') as f:
            for k, v in self.items():
                if k != 'filename':
                    f.write(f"{k} = {repr(v)}\n")

    def merge_from_dict(self, options):
        for k, v in options.items():
            self[k] = v


class DictAction(argparse.Action):
    """Argparse action for parsing key=value pairs into a dictionary."""
    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for item in values:
            if '=' in item:
                k, v = item.split('=', 1)
                try:
                    v = ast.literal_eval(v)
                except Exception:
                    pass
                options[k] = v
        setattr(namespace, self.dest, options)


def mkdir_or_exist(dir_name):
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def setup_mmcv_mock():
    """Register dummy mmcv modules in sys.modules so torch.load / unpickler succeeds."""
    if 'mmcv' not in sys.modules:
        m = types.ModuleType('mmcv')
        m.utils = types.ModuleType('mmcv.utils')
        m.utils.config = types.ModuleType('mmcv.utils.config')
        m.utils.config.ConfigDict = ConfigDict
        m.utils.config.Config = Config
        m.Config = Config
        m.ConfigDict = ConfigDict
        m.DictAction = DictAction
        m.mkdir_or_exist = mkdir_or_exist
        sys.modules['mmcv'] = m
        sys.modules['mmcv.utils'] = m.utils
        sys.modules['mmcv.utils.config'] = m.utils.config


# Auto-register on import
setup_mmcv_mock()
