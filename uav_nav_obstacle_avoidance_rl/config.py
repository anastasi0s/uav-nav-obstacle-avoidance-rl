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

# LOGGING ------------------------------------------------------------------------------------------------- #
LOG_LEVEL = "INFO"  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_FILE = Path("run.log")

# set up logging with both Rich console and file handlers
console = Console()
install(show_locals=True)  # get any traceback rendered with Rich

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        ),
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
            mode="a",  # append mode
        ),
    ],
)

# get logger instance
logger = logging.getLogger(__name__)

# log the project root path
# logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")
