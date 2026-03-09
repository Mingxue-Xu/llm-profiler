from typing import Union, Optional, AnyStr, Tuple, Dict

RESULT_TYPES = ["ADD", "MULT", "FLOPS"]

class Result:
    model_id = None
    model_layers = None
    add_ops = 0
    mul_ops = 0

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.model_id is None:
            raise AttributeError("Model name should be given.")

    def get_flops(self) -> int:
        return self.add_ops + self.mul_ops

    def get_add_ops(self) -> int:
        return self.add_ops

    def get_mul_ops(self) -> int:
        return self.mul_ops

    def get_display_str(self, cmd:str) -> str:
        assert cmd in RESULT_TYPES, "Unknown display type."
        match cmd:
            case "ADD": return f"ADD\t{self.get_add_ops()}"
            case "MULT": return f"MUL\t{self.get_mul_ops()}"
            case "FLOPS": return f"FLOPS\t{self.get_flops()}"



