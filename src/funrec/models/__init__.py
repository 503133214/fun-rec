"""模型定义"""

import importlib

from .utils import (
    build_input_layer,
    build_embedding_table_dict,
    build_group_feature_embedding_table_dict,
    concat_group_embedding,
    parse_group_feature_columns,
)

# 模型构建函数按需导入（PEP 562），避免导入 funrec.models 时加载全部模型
_LAZY_BUILDERS = {
    "build_dssm_model": "dssm",
    "build_fm_model": "fm",
    "build_afm_model": "afm",
    "build_nfm_model": "nfm",
    "build_pnn_model": "pnn",
    "build_fibinet_model": "fibinet",
    "build_deepfm_model": "deepfm",
    "build_dcn_model": "dcn",
    "build_xdeepfm_model": "xdeepfm",
    "build_autoint_model": "autoint",
    "build_din_model": "din",
    "build_dien_model": "dien",
    "build_dsin_model": "dsin",
    "build_fm_recall_model": "fm_recall",
    "build_funksvd_model": "funksvd",
    "build_biassvd_model": "biassvd",
    "build_sdm_model": "sdm",
    "build_wide_deep_model": "wide_deep",
    "build_apg_model": "apg",
    "build_m2m_model": "m2m",
    "build_prm_model": "prm",
    "build_item2vec_model": "item2vec",
}


def __getattr__(name):
    if name in _LAZY_BUILDERS:
        module = importlib.import_module(f".{_LAZY_BUILDERS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_dssm_model",
    "build_fm_model",
    "build_afm_model",
    "build_nfm_model",
    "build_pnn_model",
    "build_fibinet_model",
    "build_deepfm_model",
    "build_dcn_model",
    "build_xdeepfm_model",
    "build_autoint_model",
    "build_din_model",
    "build_dien_model",
    "build_dsin_model",
    "build_fm_recall_model",
    "build_funksvd_model",
    "build_biassvd_model",
    "build_sdm_model",
    "build_wide_deep_model",
    "build_apg_model",
    "build_m2m_model",
    "build_prm_model",
    "build_item2vec_model",
    "build_input_layer",
    "build_embedding_table_dict",
    "build_group_feature_embedding_table_dict",
    "concat_group_embedding",
    "parse_group_feature_columns",
]
