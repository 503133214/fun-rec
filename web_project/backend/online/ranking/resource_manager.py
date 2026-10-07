"""
精排模型资源管理器

负责从本地共享目录加载精排模型和词表。
实现单例模式以避免重复加载模型。

注意：PyTorch / funrec 采用懒加载方式导入，以加快服务启动速度。
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder

from app.config import settings

logger = logging.getLogger(__name__)


def _get_funrec_load_model():
    """
    懒加载 funrec 的 PyTorch 模型加载函数，并设置推理设备。

    funrec 的 load_model 会根据保存的构建函数路径、特征列和模型参数重建模型并加载参数，
    因此无需额外注册自定义层。若 funrec 不在 PYTHONPATH 中，
    则回退为将仓库根目录下的 src/ 加入 sys.path（与离线训练脚本一致）。
    推理设备由环境变量 MODEL_DEVICE 指定（默认 cpu，在线服务通常无 GPU）。
    """
    try:
        import funrec  # noqa: F401
    except ImportError:
        # online/<模块>/resource_manager.py -> 仓库根目录 (web_project 的上一级)
        parents = Path(__file__).resolve().parents
        src_dir = parents[4] / "src" if len(parents) > 4 else None
        if src_dir is not None and src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.append(str(src_dir))
    from funrec.models.base import load_model, set_device

    set_device(os.getenv("MODEL_DEVICE", "cpu"))
    return load_model


class RankingResourceManager:
    """
    精排模型的单例资源管理器。
    
    加载并缓存：
    - DeepFM 精排模型
    - 特征词表（用于编码）
    - 特征字典（词表大小）
    
    注意：资源采用懒加载方式，在首次访问时才加载，以避免启动时导入 PyTorch 的开销。
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RankingResourceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        # 模型部署目录配置
        self.deploy_dir = Path(os.getenv("MODEL_DEPLOY_DIR", "/app/tmp/web_project/deployed_models"))
        
        # 资源（ranking_model 类型为 Any 以避免模块级别的 torch 导入）
        self.ranking_model: Optional[Any] = None
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_dict: Dict[str, int] = {}
        self.model_config: Dict[str, Any] = {}
        
        # DeepFM 所需的特征名称
        self.user_features = ["user_id", "gender", "age", "occupation", "zip_code"]
        self.item_features = ["movie_id", "genres", "isAdult", "startYear"]
        self.all_features = self.user_features + self.item_features
        
        # 跟踪加载状态 - 资源在首次访问时懒加载
        self._resources_loaded = False
        self._initialized = True
        
    def _ensure_resources_loaded(self):
        """在首次访问时懒加载资源。"""
        if not self._resources_loaded:
            self._load_resources()
            self._resources_loaded = True

    def _load_resources(self):
        """从本地目录加载所有精排资源。"""
        logger.info(f"从本地目录加载精排资源: {self.deploy_dir}")
        
        self._load_vocab()
        self._load_feature_dict()
        self._load_model_config()
        self._load_ranking_model()
        
        logger.info("精排资源加载完成。")

    def _load_vocab(self):
        """加载特征编码用的词表字典。"""
        try:
            vocab_path = self.deploy_dir / "ranking" / "vocab_dict.pkl"
            logger.info(f"  加载精排 vocab_dict.pkl: {vocab_path}")
            
            with open(vocab_path, "rb") as f:
                vocab_dict = pickle.load(f)
            
            # 为每个特征构建 LabelEncoder
            for feature_name, classes in vocab_dict.items():
                le = LabelEncoder()
                le.classes_ = classes if isinstance(classes, np.ndarray) else np.array(classes)
                self.encoders[feature_name] = le
                
            logger.info(f"    ✓ 已加载特征词表: {list(self.encoders.keys())}")
            
        except Exception as e:
            logger.error(f"  ✗ 加载精排词表时出错: {e}")

    def _load_feature_dict(self):
        """加载特征字典（词表大小）。"""
        try:
            feat_path = self.deploy_dir / "ranking" / "feature_dict.pkl"
            logger.info(f"  加载精排 feature_dict.pkl: {feat_path}")
            
            with open(feat_path, "rb") as f:
                self.feature_dict = pickle.load(f)
            logger.info(f"    ✓ 特征词表大小: {self.feature_dict}")
            
        except Exception as e:
            logger.error(f"  ✗ 加载特征字典时出错: {e}")

    def _load_model_config(self):
        """加载模型配置。"""
        try:
            config_path = self.deploy_dir / "ranking" / "model_config.pkl"
            logger.info(f"  加载精排 model_config.pkl: {config_path}")
            
            with open(config_path, "rb") as f:
                self.model_config = pickle.load(f)
            logger.info(f"    ✓ 模型配置已加载")
            
        except Exception as e:
            logger.warning(f"  ⚠ 未找到模型配置（可选）: {e}")

    def _load_ranking_model(self):
        """从本地目录加载 DeepFM 精排模型。"""
        try:
            # 检查当前激活的版本
            active_json_path = self.deploy_dir / "model" / "ranking" / "active.json"
            try:
                with open(active_json_path, "r") as f:
                    version_info = json.load(f)
                model_rel_path = version_info.get("path")
                logger.info(f"  从 {model_rel_path} 加载精排模型...")
            except Exception:
                logger.warning("  无法找到激活的精排模型版本，尝试使用默认路径")
                model_rel_path = "model/ranking/v1/ranking_model.pt"

            if model_rel_path:
                model_path = self.deploy_dir / model_rel_path
                
                if not model_path.exists():
                    logger.error(f"  ✗ 模型文件不存在: {model_path}")
                    return
                if model_path.is_dir():
                    logger.error(f"  ✗ {model_path} 是目录（旧版 TensorFlow SavedModel?），请重新训练并部署 PyTorch 模型 (.pt)")
                    return
                
                logger.info(f"    从 {model_path} 加载 PyTorch 模型...")
                load_model = _get_funrec_load_model()
                # load_model 返回 eval 模式的 FunRecModel，.predict(dict) 输出 numpy 数组
                self.ranking_model = load_model(str(model_path), map_location="cpu")
                logger.info("    ✓ 精排模型加载成功")
            else:
                logger.warning("  未找到激活的精排模型路径。")
                
        except Exception as e:
            logger.error(f"  ✗ 加载精排模型时出错: {e}")
            import traceback
            traceback.print_exc()

    def encode_feature(self, feature_name: str, value: Any) -> int:
        """
        编码单个特征值。
        
        Args:
            feature_name: 特征名称
            value: 原始特征值
            
        Returns:
            编码后的整数值（从 1 开始，0 表示未知）
        """
        self._ensure_resources_loaded()
        
        if value is None:
            return 0
            
        encoder = self.encoders.get(feature_name)
        if encoder is None:
            return 0
            
        try:
            # 处理类型转换
            if isinstance(value, (int, float)) and encoder.classes_.dtype.kind in ('U', 'S', 'O'):
                value = str(int(value) if isinstance(value, float) else value)
            elif isinstance(value, str) and encoder.classes_.dtype.kind in ('i', 'f'):
                value = int(value) if value.isdigit() else 0
                
            # 检查值是否存在于词表中
            if value in encoder.classes_:
                return int(encoder.transform([value])[0]) + 1  # 从 1 开始编号
            else:
                return 0  # 未知值
        except Exception as e:
            logger.debug(f"编码 {feature_name}={value} 时出错: {e}")
            return 0

    @property
    def is_ready(self) -> bool:
        """检查资源是否已加载并就绪。"""
        self._ensure_resources_loaded()
        return self.ranking_model is not None and len(self.encoders) > 0
