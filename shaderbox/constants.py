"""Constants used throughout the shaderbox application."""

from importlib.resources import files
from pathlib import Path

# Resource directories
RESOURCES_DIR = Path(str(files("shaderbox.resources")))
NODE_EXAMPLES_DIR = RESOURCES_DIR / "node_examples"
SHADER_LIB_SEED_DIR = RESOURCES_DIR / "shader_lib"

# Authored display order for the examples browser — filesystem ctime isn't preserved
# through git/zip/bundle. Examples not listed sort last. The first is the procedural
# starter cloned by "New node" and seeded into an empty project on first run.
EXAMPLE_ORDER = [
    "53724dbd-8efb-4c09-8c7d-28d626a066e7",  # UV Mango
    "73ea2431-13f6-41e4-b923-04d846b678b0",  # Media Input
    "f90f5ff9-29c6-4bcf-aee7-090f20542353",  # Text Rendering
    "0b0d16bb-f014-4a85-b155-6be74c33eded",  # Fire
    "8d454b7b-bd48-49dc-aebe-58b9e31cfc28",  # Night City
]
STARTER_EXAMPLE_ID = EXAMPLE_ORDER[0]

# Default file paths
DEFAULT_VS_FILE_PATH = RESOURCES_DIR / "shaders" / "default.vert.glsl"
DEFAULT_FS_FILE_PATH = RESOURCES_DIR / "shaders" / "default.frag.glsl"
DEFAULT_IMAGE_FILE_PATH = RESOURCES_DIR / "textures" / "default.jpeg"

# Canvas and texture sizes
DEFAULT_CANVAS_SIZE = (64, 64)

# Video encoding settings
MP4_CRF_VALUES = [33, 28, 23, 18]  # Quality levels: 0=lowest, 3=highest
MP4_PRESETS = ["ultrafast", "fast", "medium", "slow"]
WEBM_CRF_VALUES = [50, 40, 30, 20]
WEBM_CPU_USED_VALUES = [5, 4, 3, 2]

# Video resolution alignment (for codec compatibility)
VIDEO_RESOLUTION_ALIGNMENT = 16

# Temporal smoothing defaults
DEFAULT_TEMPORAL_WINDOW_SIZE = 5
DEFAULT_TEMPORAL_SIGMA = 1.0
DEFAULT_TEMPORAL_QUALITY = 2

# Default video settings
DEFAULT_FPS = 30

# File extensions
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
VIDEO_EXTENSIONS = [".mp4", ".webm", ".mov"]
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
GLSL_EXTENSIONS = [".glsl", ".frag"]

# Directory names for node loading
MEDIA_DIR_NAME = "media"
TEXTURES_DIR_NAME = "textures"

# Vertex data for full-screen quad
FULLSCREEN_QUAD_VERTICES = [
    -1.0,
    -1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
]
