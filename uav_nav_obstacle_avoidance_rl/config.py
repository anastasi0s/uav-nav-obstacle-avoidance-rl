from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# PATHS ------------------------------------------------------------------------------------------------- #
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
TRAIN_RESULTS = REPORTS_DIR / "training_results"
EVAL_RESULTS = REPORTS_DIR / "evaluation_results"
FIGURES_DIR = REPORTS_DIR / "figures"
VIDEOS_DIR = REPORTS_DIR / "videos"


# ENVIRONMENT --------------------------------------------------------------------------------------------- #


# TRAINING ------------------------------------------------------------------------------------------------ #


# EVALUATION ------------------------------------------------------------------------------------------------ #


# LOGGING ------------------------------------------------------------------------------------------------- #
LOG_LEVEL = "INFO"  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_FILE = Path("logs.log")

# init Loguru
logger.remove()  # remove default Loguru handlers

# define a Loguru format string
format_string = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        level=LOG_LEVEL,
        format=format_string,
        colorize=True,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
except ModuleNotFoundError:
    pass

# # add console sink
# logger.add(sys.stdout, level=config.LOG_LEVEL, format=format_string, enqueue=True, backtrace=True, diagnose=True)

# add file sink with rotation
logger.add(
    LOG_FILE,
    level=LOG_LEVEL,
    format=format_string,
    rotation="10 MB",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)
