import os
import yaml
from typing import Dict, Any, Optional


class ConfigManager:
    _instance = None
    _config = {}
    
    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance
    
    def _load_config(self, config_path: str = None):
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
        
        if config_path and os.path.exists(config_path):
            self._config = self._read_yaml(config_path)
        elif os.path.exists(default_path):
            self._config = self._read_yaml(default_path)
        else:
            self._config = self._get_default_config()
            self._save_yaml(default_path, self._config)
    
    def _read_yaml(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"读取配置文件失败: {e}")
            return self._get_default_config()
    
    def _save_yaml(self, path: str, config: Dict[str, Any]):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'model': {
                'neuro_nli_model': 'IDEA-CCNL/Erlangshen-Roberta-110M-NLI',
                'fine_tuned_model_path': 'models/fine_tuned_nli',
                'device': 'cpu',
                'use_fine_tuned': True
            },
            'rule_engine': {
                'high_confidence_threshold': 0.70,
                'low_confidence_threshold': 0.50,
                'enable_negation_coverage': True,
                'enable_inferential_conflict': True,
                'enable_laterality_conflict': True
            },
            'nli': {
                'confidence_threshold': 0.70,
                'ambiguous_lower_bound': 0.45,
                'ambiguous_upper_bound': 0.65,
                'enable_neuro_nli': True,
                'enable_rule_nli_fallback': True
            },
            'log': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_path': 'logs/radiology_checker.log',
                'max_file_size_mb': 10,
                'backup_count': 5
            },
            'parser': {
                'enable_uncertainty_detection': True,
                'enable_laterality_detection': True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def get_model_config(self) -> Dict[str, Any]:
        return self._config.get('model', {})
    
    def get_rule_engine_config(self) -> Dict[str, Any]:
        return self._config.get('rule_engine', {})
    
    def get_nli_config(self) -> Dict[str, Any]:
        return self._config.get('nli', {})
    
    def get_log_config(self) -> Dict[str, Any]:
        return self._config.get('log', {})
    
    def get_parser_config(self) -> Dict[str, Any]:
        return self._config.get('parser', {})
    
    def reload(self, config_path: str = None):
        self._load_config(config_path)
    
    def to_dict(self) -> Dict[str, Any]:
        return self._config.copy()


def get_config() -> ConfigManager:
    return ConfigManager()
