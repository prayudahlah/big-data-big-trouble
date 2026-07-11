from modules.models.factory import TrashClassifier

__all__ = ["TrashClassifier"]

try:
    from modules.models.clip_adapter import CLIPAdapter

    __all__.append("CLIPAdapter")
except ImportError:
    pass
