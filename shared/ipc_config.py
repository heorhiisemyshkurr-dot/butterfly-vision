"""
IPC protocol definitions — Python equivalent of ipc_config.h.
Must be kept in sync with the C++ header at all times.
"""
import ctypes

# Shared Memory / Semaphore names
BUTTERFLY_FRAME_NAME   = "Local\\ButterflyFrame"
BUTTERFLY_READY_NAME   = "Local\\ButterflyReady"
BUTTERFLY_CONFIG_NAME  = "Local\\ButterflyConfig"

# Geometry limits
BUTTERFLY_MAX_WIDTH    = 3840
BUTTERFLY_MAX_HEIGHT   = 2160
BUTTERFLY_BYTES_PER_PX = 4   # BGRA
BUTTERFLY_FRAME_PIXELS = BUTTERFLY_MAX_WIDTH * BUTTERFLY_MAX_HEIGHT
BUTTERFLY_FRAME_BYTES  = BUTTERFLY_FRAME_PIXELS * BUTTERFLY_BYTES_PER_PX

# Magic constants (version guards)
BUTTERFLY_FRAME_MAGIC  = 0xBF190001  # same bits as C++ (note: 'Y' hex = 0x59)
BUTTERFLY_CONFIG_MAGIC = 0xBF19CF60


class ButterflyFrameHeader(ctypes.Structure):
    """
    Matches _ButterflyFrameHeader in ipc_config.h exactly.
    Total size: 32 bytes (packed).
    """
    _pack_ = 1
    _fields_ = [
        ("magic",        ctypes.c_uint32),
        ("frame_id",     ctypes.c_uint32),
        ("width",        ctypes.c_uint32),
        ("height",       ctypes.c_uint32),
        ("stride",       ctypes.c_uint32),
        ("timestamp_ms", ctypes.c_uint32),
        ("reserved",     ctypes.c_uint8 * 8),
    ]


class ButterflyConfig(ctypes.Structure):
    """
    Matches _ButterflyConfig in ipc_config.h exactly.
    Total size: 32 bytes (packed).
    """
    _pack_ = 1
    _fields_ = [
        ("magic",    ctypes.c_uint32),
        ("center_x", ctypes.c_int32),
        ("center_y", ctypes.c_int32),
        ("radius",   ctypes.c_int32),
        ("enabled",  ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 12),
    ]


HEADER_SIZE     = ctypes.sizeof(ButterflyFrameHeader)   # 32 bytes
PIXEL_OFFSET    = HEADER_SIZE
SHM_TOTAL_SIZE  = HEADER_SIZE + BUTTERFLY_FRAME_BYTES
CFG_SIZE        = ctypes.sizeof(ButterflyConfig)         # 32 bytes
