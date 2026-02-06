import sys
from real_utils import collect_data,collect_energy_data, insert_to_db_energy, calculate_true_consumption, import_dataframes_to_db, collect_comfort_data, insert_to_db_comfort, get_historic_complete_data
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

import os

load_dotenv()  # loads .env into environment variables

def get_time_range_minutes(minutes_back: int = 30) -> tuple[str, str]:
    """Get time range for last N minutes (default 30)"""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=minutes_back)

    def format_time(dt):
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    return format_time(start_time), format_time(end_time)

def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env variable: {name}")
    return value

def main():
    try:
        # Get last 30 minutes of data (with overlap for upsert)
        start_time, end_time = get_time_range_minutes(30)
        locations = [
            "Dimarxeio_sunedriaston",
            "Dimarxeio_rack",
            "Oikeia_antidimarxou",
            "Oikeia_xatziara",
            "Super_market",
            "Cafeteria",
        ]
        locations_for_history = [
            "Dimarxeio_sunedriaston",
            "Dimarxeio_rack",
            "Oikeia_antidimarxou",
            "Oikeia_xatziara",
        ]
        comfort_rows_total = 0
        for loc in locations:
            tin = get_env(f"{loc}_tin")
            rh = get_env(f"{loc}_rh")
            # if loc in locations_for_history:
            #     df_hist = get_historic_complete_data(tin, loc)
            df_tin = collect_comfort_data(loc, tin, start_time, end_time)
            df_rh = collect_comfort_data(loc, rh, start_time, end_time)
            if loc == 'Dimarxeio_sunedriaston':
                con_dimarxeio = get_env(f"{loc}_consumption")
                df_cons = collect_energy_data(loc, con_dimarxeio, start_time, end_time)
                # print('== LOCATION == ', loc, '\n')
                print(df_cons)
                # energy_rows = insert_to_db_energy(df_cons)
           
            print('== LOCATION == ', loc, '\n')
            print(df_tin, df_rh)
            # comfort_rows = insert_to_db_comfort(df_tin, df_rh, loc)
            # comfort_rows_total += comfort_rows
        
        loc_tin = "Dimarxeio_sunedriaston_tin"
        loc = "Dimarxeio_sunedriaston"
        loc_energy = "Dimarxeio_sunedriaston_energy"
        # df_cons = get_historic_complete_data(con_dimarxeio, loc_energy)
        # df_tin = get_historic_complete_data(tin, loc_tin)
        # df_rh = get_historic_complete_data(rh, loc_rh)
        # Collect data
        pv_prod = collect_data("pv_prod", start_time, end_time)
        tc = calculate_true_consumption(start_time, end_time)

        # # # Import to database
        prod_rows, cons_rows = import_dataframes_to_db(pv_prod, tc)
        print(
            f"✅ Import completed: {prod_rows} production, {cons_rows} consumption rows, {comfort_rows_total} comfort_rows")

    except Exception as e:
        print(f"❌ Cronjob failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
