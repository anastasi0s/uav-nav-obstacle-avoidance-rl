import logging
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# load environment variables from .env file if it exists
load_dotenv()

# random seed as a consatant for experiment reproducibility
RANDOM_SEED = 319

# PATHS ------------------------------------------------------------------------------------------------- #
PROJ_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = PROJ_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = REPORTS_DIR / "run.log"

EXP_CONFIG_PATH = PROJ_ROOT / "uav_nav_obstacle_avoidance_rl" / "modeling" / "config-defaults.yaml"
SWEEP_CONFIG_PATH = PROJ_ROOT / "uav_nav_obstacle_avoidance_rl" / "modeling" / "sweep-config.yaml"


# LOGGING ------------------------------------------------------------------------------------------------- #
LOG_LEVEL = "INFO"  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

# set up logging with both Rich console and file handlers
console = Console()
install(show_locals=True)  # get any traceback rendered with Rich

LOGGER_NAME = "uav_nav_rl"

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(getattr(logging, LOG_LEVEL))
logger.propagate = False  # don't let root logger interfere

# Rich console handler
logger.addHandler(
    RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
)

# File handler
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# log the project root path
# logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")
