"""
召回模型资源管理器

负责从本地共享目录加载召回模型和词表。
实现单例模式以避免重复加载模型。

注意：PyTorch / funrec 采用懒加载方式导入，以加快服务启动速度。
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from app.config import settings

# 配置日志
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


class RecallResourceManager:
    """
    召回模型的单例资源管理器。
    
    注意：资源采用懒加载方式，在首次访问时才加载，以避免启动时导入 PyTorch 的开销。
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RecallResourceManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        # 模型部署目录配置
        self.deploy_dir = Path(os.getenv("MODEL_DEPLOY_DIR", "/app/tmp/web_project/deployed_models"))
        
        self.user_model = None
        self.encoders = {}
        self.all_movie_ids = []
        self.item_embedding_matrix = None
        self.movie_genre_map = {}
        
        # 跟踪加载状态 - 资源在首次访问时懒加载
        self._resources_loaded = False
        self.initialized = True
        
    def _ensure_resources_loaded(self):
        """在首次访问时懒加载资源。"""
        if not self._resources_loaded:
            self.load_resources()
            self._resources_loaded = True

    def load_resources(self):
        logger.info(f"从本地目录加载召回资源: {self.deploy_dir}")
        
        # 1. 加载词表
        try:
            vocab_path = self.deploy_dir / "vocab_dict.pkl"
            logger.info(f"加载 vocab_dict.pkl: {vocab_path}")
            
            with open(vocab_path, "rb") as f:
                vocab_dict = pickle.load(f)
            
            for feature_name, classes in vocab_dict.items():
                le = LabelEncoder()
                le.classes_ = classes if isinstance(classes, np.ndarray) else np.array(classes)
                self.encoders[feature_name] = le
                
                if feature_name == "movie_id":
                    self.all_movie_ids = list(le.classes_)
            logger.info("  -> 词表加载完成。")
        except Exception as e:
            logger.error(f"加载词表时出错: {e}")

        # 2. 加载物品 Embedding
        try:
            emb_path = self.deploy_dir / "item_embeddings.npy"
            logger.info(f"加载物品 Embedding: {emb_path}")
            
            self.item_embeddings = np.load(emb_path)
            
            # 设置带填充位的 Embedding 矩阵
            dim = self.item_embeddings.shape[1]
            self.item_embedding_matrix = np.zeros((self.item_embeddings.shape[0] + 1, dim), dtype=np.float32)
            self.item_embedding_matrix[1:] = self.item_embeddings
            
            logger.info(f"  -> 物品 Embedding 加载完成。形状: {self.item_embedding_matrix.shape}")
            
        except Exception as e:
            logger.error(f"加载物品 Embedding 时出错: {e}")
            self.item_embedding_matrix = None

        # 3. 加载用户模型
        try:
            # 检查当前激活的版本
            active_json_path = self.deploy_dir / "model" / "user_recall" / "active.json"
            try:
                with open(active_json_path, "r") as f:
                    version_info = json.load(f)
                model_rel_path = version_info.get("path")
            except:
                logger.warning("无法找到激活的模型版本，尝试使用默认路径")
                model_rel_path = "model/user_recall/v1/user_model.pt"

            if model_rel_path:
                model_path = self.deploy_dir / model_rel_path
                
                if not model_path.exists():
                    logger.error(f"模型文件不存在: {model_path}")
                    return
                if model_path.is_dir():
                    logger.error(f"{model_path} 是目录（旧版 TensorFlow SavedModel?），请重新训练并部署 PyTorch 模型 (.pt)")
                    return
                
                logger.info(f"从 {model_path} 加载用户模型...")
                load_model = _get_funrec_load_model()
                # 保存时 tower='user_model'，load_model 直接返回用户塔子模型（eval 模式）
                self.user_model = load_model(str(model_path), map_location="cpu")
                logger.info(f"  -> 用户模型加载完成。")
            else:
                logger.warning("未找到激活的模型路径。")
        except Exception as e:
             logger.error(f"加载用户模型时出错: {e}")

    def set_movie_genre_map(self, movie_db_objects):
        """从数据库对象填充电影类型映射表"""
        genre_le = self.encoders.get("genres")
        if not genre_le:
            return

        for movie in movie_db_objects:
            raw_mid = movie.movie_id 
            try:
                if hasattr(movie, 'genres') and movie.genres:
                    # 处理 genres 为字符串或列表的情况
                    genres = movie.genres if isinstance(movie.genres, list) else str(movie.genres).split('|')
                    # 过滤出编码器中存在的类型
                    valid_genres = [g for g in genres if g in genre_le.classes_]
                    if valid_genres:
                        encoded_genres = genre_le.transform(valid_genres) + 1
                        self.movie_genre_map[raw_mid] = encoded_genres
                    else:
                        self.movie_genre_map[raw_mid] = []
                else:
                     self.movie_genre_map[raw_mid] = []
            except Exception as e:
                 logger.debug(f"映射电影 {raw_mid} 的类型时出错: {e}")
                 self.movie_genre_map[raw_mid] = []
