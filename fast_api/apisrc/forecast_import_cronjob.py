import logging
import sys
from forecast_utils import run_all_forecasts

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # Log to the same file as the main module for unified logging
        logging.FileHandler('./forecast_system.log'),
        # Also print to standard output, which cron can redirect to a file
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    """
    Main entry point for the daily cron job.
    Executes the PV and consumption forecast processes.
    """
    logger.info("=========================================================")
    logger.info("CRON JOB: Starting daily forecast generation.")
    logger.info("=========================================================")

    try:
        # run_all_forecasts() executes both PV and consumption forecasts
        pv_rows, consumption_rows = run_all_forecasts()

        logger.info(f"SCRIPT SUMMARY - PV forecast rows affected: {pv_rows}")
        logger.info(
            f"SCRIPT SUMMARY - Consumption forecast rows affected: {consumption_rows}")
        logger.info("---------------------------------------------------------")
        logger.info(
            "CRON JOB: Daily forecast generation finished successfully.")
        logger.info("---------------------------------------------------------")

    except Exception as e:
        logger.critical(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.critical(f"CRON JOB: An unhandled error occurred: {e}")
        logger.critical(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Exit with a non-zero status code to indicate failure, which cron can detect
        sys.exit(1)

    # Exit with a zero status code for success
    sys.exit(0)
