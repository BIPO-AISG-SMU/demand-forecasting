import bipo as bipo
import bipo_fastapi as bipo_fapi


PRED_MODEL = bipo.modeling.utils.load_model(
    bipo_fapi.config.SETTINGS.PRED_MODEL_PATH)
