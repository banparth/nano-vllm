import enum


class CUDAGraphMode(enum.Enum):
    FULL = "full"
    PIECEWISE = "piecewise"
    BREAKABLE = "breakable"
