"""Standalone price-alert worker — sends email/SMS alerts without the Streamlit app running.

    python alert_worker.py            # run forever (checks every ALERT_CHECK_MINUTES)
    python alert_worker.py --once     # one refresh + check, then exit (for Task Scheduler / cron)

Reads the same .env and data/ files as the app. If you run this, set
BACKGROUND_ALERTS=0 in .env so the app's built-in worker doesn't also send alerts.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()

from data.alert_job import _stamp, check_minutes, refresh_minutes, run_cycle, worker_loop  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="refresh prices (if the market is open), check once, exit")
    args = parser.parse_args()

    if args.once:
        messages, _ = run_cycle(refresh=True)
        for m in messages or ["No new alerts."]:
            print(f"{_stamp()}  {m}")
        return

    print(f"{_stamp()}  Alert worker started — checking every {check_minutes():g} min, "
          f"refreshing prices every {refresh_minutes():g} min during NSE hours. Ctrl+C to stop.")
    try:
        worker_loop({}, on_message=print)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
