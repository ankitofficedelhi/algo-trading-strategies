import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, time

cwd = Path.cwd()
SLIPAGES = {
    "nifty": 0.01,
    "banknifty": 0.0125,
    "finnifty": 0.01,
    "midcpnifty": 0.0125,
    "sensex": 0.0125,
    "bankex": 0.0125,
}
STEPS = {
    "nifty": 1000,
    "banknifty": 5000,
    "finnifty": 1000,
    "midcpnifty": 1000,
    "sensex": 5000,
    "bankex": 5000,
    "spxw": 500,
    "xsp": 50,
}
PREFIX = {
    "nifty": "Nifty",
    "banknifty": "BN",
    "finnifty": "FN",
    "midcpnifty": "MCN",
    "sensex": "SX",
    "bankex": "BX",
}

def get_gap(options):
    try:
        strike = options.scrip.str[:-2].astype(float).unique()
        strike.sort()
        differences = np.diff(strike)
        min_gap = float(differences.min())
        min_gap = int(min_gap) if min_gap.is_integer() else min_gap
        return min_gap
    except Exception as e:
        print(e)
        return None

def get_file_date(file: Path) -> datetime:
    """
    Extract date from filename prefix: YYYY-MM-DD_xxx.pkl
    """
    try:
        date_str = file.stem.split("_")[0]  # take '2019-03-07'
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None

def load_files_in_date_range(
    opt_file_path: Path, fut_file_path: Path, start_date: datetime, end_date: datetime
) -> tuple[list[Path], list[Path]]:
    options_dict = {}
    for file in opt_file_path.iterdir():
        if file.is_file():
            file_date = get_file_date(file)
            if file_date and start_date <= file_date <= end_date:
                options_dict[file_date.date()] = file

    futures_dict = {}
    for file in fut_file_path.iterdir():
        if file.is_file():
            file_date = get_file_date(file)
            if file_date and start_date <= file_date <= end_date:
                futures_dict[file_date.date()] = file

    common_dates = sorted(set(options_dict.keys()) & set(futures_dict.keys()))
    options_list = [options_dict[d] for d in common_dates]
    futures_list = [futures_dict[d] for d in common_dates]

    return options_list, futures_list

def group_files_by_week(
    nifty_future_list, nifty_options_list, index: str
) -> list[list]:
    """Group files by expiry week based on DTE decreasing sequence."""
    index = index.upper()
    dtes = pd.read_csv(r"C:\PICKLE\DTE.csv", parse_dates=["Date"]).set_index("Date")
    # dtes.index = pd.to_datetime(dtes.index, format="%d-%m-%Y").date()
    dtes.index = [pd.to_datetime(dt, format="%d-%m-%Y").date() for dt in dtes.index]
    if index not in dtes.columns:
        raise ValueError(f"DTE column for index '{index}' not found in DTE.csv")

    weekly_groups = []
    current_group = []
    last_dte = None

    for fut_path, opt_path in zip(nifty_future_list, nifty_options_list):

        file_dt = get_file_date(opt_path)
        dte = dtes.loc[file_dt.date(), index]
        if dte is None:
            continue
        if last_dte is None or dte < last_dte:
            current_group.append((fut_path, opt_path))
        else:
            if current_group:
                weekly_groups.append(current_group)
            current_group = [(fut_path, opt_path)]
        last_dte = dte
    if current_group:
        weekly_groups.append(current_group)
    return weekly_groups

def get_straddle_strike(
    opt: pd.DataFrame,
    fut: pd.DataFrame,
    start_dt: pd.Timedelta,
    end_dt: pd.Timedelta,
    gap: int = 50,
    sd: int = 0,
    SDroundoff: bool = False,
):
    # Get the date from the futures data
    current_date = fut.index[0].date()
    # Convert time objects to datetime
    start_datetime = (
        pd.Timestamp.combine(current_date, start_dt)
        if isinstance(start_dt, time)
        else start_dt
    )
    end_datetime = (
        pd.Timestamp.combine(current_date, end_dt) - pd.Timedelta(minutes=15)
        if isinstance(end_dt, time)
        else end_dt - pd.Timedelta(minutes=15)
    )

    valid_times = fut.loc[start_datetime:end_datetime].index
    for current_dt in valid_times:
        try:
            future_price = fut.loc[current_dt, "close"]
            round_future_price = round(future_price / gap) * gap
            ce_scrip, pe_scrip = f"{round_future_price}CE", f"{round_future_price}PE"
            ce_price, pe_price = (
                opt[
                    (opt["date_time"] == current_dt) & (opt["scrip"] == ce_scrip)
                ].close.iloc[0],
                opt[
                    (opt["date_time"] == current_dt) & (opt["scrip"] == pe_scrip)
                ].close.iloc[0],
            )
            syn_future = ce_price - pe_price + round_future_price
            round_syn_future = round(syn_future / gap) * gap
            ce_scrip_list = [
                f"{round_syn_future}CE",
                f"{round_syn_future+gap}CE",
                f"{round_syn_future-gap}CE",
            ]
            pe_scrip_list = [
                f"{round_syn_future}PE",
                f"{round_syn_future+gap}PE",
                f"{round_syn_future-gap}PE",
            ]
            scrip_index, min_value = 0, float("inf")
            for i in range(3):
                try:
                    ce_price = opt[
                        (opt.index == current_dt) & (opt["scrip"] == ce_scrip_list[i])
                    ].close.iloc[0]
                    pe_price = opt[
                        (opt.index == current_dt) & (opt["scrip"] == pe_scrip_list[i])
                    ].close.iloc[0]

                    diff = abs(ce_price - pe_price)
                    if min_value > diff:
                        min_value = diff
                        scrip_index = i
                except:
                    pass
            ce_scrip, pe_scrip = ce_scrip_list[scrip_index], pe_scrip_list[scrip_index]
            ce_price, pe_price = (
                opt[(opt.index == current_dt) & (opt["scrip"] == ce_scrip)].close.iloc[
                    0
                ],
                opt[(opt.index == current_dt) & (opt["scrip"] == pe_scrip)].close.iloc[
                    0
                ],
            )
            sd_range = 0
            if sd != 0:
                sd_range = (ce_price + pe_price) * sd

                if SDroundoff:
                    sd_range = round(sd_range / gap) * gap
                else:
                    sd_range = max(gap, round(sd_range / gap) * gap)

            ce_scrip, pe_scrip = (
                f"{int(ce_scrip[:-2])+int(sd_range)}CE",
                f"{int(pe_scrip[:-2])-int(sd_range)}PE",
            )
            ce_price, pe_price = (
                opt[(opt.index == current_dt) & (opt["scrip"] == ce_scrip)].close.iloc[
                    0
                ],
                opt[(opt.index == current_dt) & (opt["scrip"] == pe_scrip)].close.iloc[
                    0
                ],
            )
            return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt
        except (TypeError, ValueError, KeyError, IndexError):
            print("error occured in get straddle strike")
            continue

    return None, None, None, None, None, None

def get_one_om(fut, future_price=None, STEP=1000):
    future_price = fut["close"].iloc[0] if future_price is None else future_price
    return (int(future_price / STEP) * STEP) / 100

def get_strangle_strike(
    opt,
    fut,
    start_time,
    end_time,
    gap=50,
    om=None,
    target=None,
    check_inverted=False,
    tf=1,
    index=None,
):
    # Get the date from the futures data
    current_date = fut.index[0].date()

    # Convert time objects to datetime
    start_datetime = pd.Timestamp.combine(current_date, start_time)
    end_datetime = pd.Timestamp.combine(current_date, end_time) - pd.Timedelta(
        minutes=15
    )

    valid_times = fut.loc[start_datetime:end_datetime].index
    for current_dt in valid_times:
        try:
            future_price = fut.loc[current_dt, "close"]
            # future price = 17510.4
            one_om = get_one_om(fut, future_price, STEP=STEPS[index.lower()])
            # one om = 170.0
            target = one_om * om if target is None else target
            # target = 68.0
            target_od = (
                opt[(opt.index == current_dt) & (opt["close"] >= target * tf)]
                .sort_values(by=["close"])
                .copy()
            )

            ce_scrip = target_od.loc[
                target_od["scrip"].str.endswith("CE"), "scrip"
            ].iloc[0]
            pe_scrip = target_od.loc[
                target_od["scrip"].str.endswith("PE"), "scrip"
            ].iloc[0]
            # ce_scrip = 17550CE
            # pe_scrip = 17400PE

            ce_scrip_list = [
                ce_scrip,
                f"{int(ce_scrip[:-2])-gap}CE",
                f"{int(ce_scrip[:-2])+gap}CE",
            ]
            pe_scrip_list = [
                pe_scrip,
                f"{int(pe_scrip[:-2])-gap}PE",
                f"{int(pe_scrip[:-2])+gap}PE",
            ]
            # ce_scrip_list = ['17550CE', '17500CE', '17600CE']
            # pe_scrip_list = ['17400PE', '17350PE', '17450PE']

            call_list_prices, put_list_prices = [], []
            for z in range(3):
                try:
                    call_list_prices.append(
                        opt[
                            (opt["date_time"] == current_dt)
                            & (opt["scrip"] == ce_scrip_list[z])
                        ]["close"].iloc[0]
                    )
                except:
                    call_list_prices.append(0)
                    # print(
                    #     "call list price is empty and my date time column is not present in your data"
                    # )
                try:
                    put_list_prices.append(
                        opt[
                            (opt["date_time"] == current_dt)
                            & (opt["scrip"] == pe_scrip_list[z])
                        ]["close"].iloc[0]
                    )
                except:
                    put_list_prices.append(0)
                    # print(
                    #     "put list price is empty and my date time column is not present in your data"
                    # )

            call, put, min_diff = call_list_prices[0], put_list_prices[0], float("inf")
            target_2, target_3 = target * 2 * tf, target * 3
            diff = abs(put - call)

            required_call, required_put = None, None
            if (put + call >= target_2) & (min_diff > diff) & (put + call <= target_3):
                min_diff = diff
                required_call, required_put = call, put

            for i in range(1, 3):
                if (
                    (min_diff > abs(put_list_prices[i] - call))
                    & (put_list_prices[i] + call >= target_2)
                    & (put_list_prices[i] + call <= target_3)
                ):
                    min_diff = abs(put_list_prices[i] - call)
                    required_call, required_put = call, put_list_prices[i]

                if (
                    (min_diff > abs(call_list_prices[i] - put))
                    & (call_list_prices[i] + put >= target_2)
                    & (call_list_prices[i] + put <= target_3)
                ):
                    min_diff = abs(call_list_prices[i] - put)
                    required_call, required_put = call_list_prices[i], put

            ce_scrip, pe_scrip = (
                ce_scrip_list[call_list_prices.index(required_call)],
                pe_scrip_list[put_list_prices.index(required_put)],
            )
            ce_price, pe_price = (
                opt[(opt["date_time"] == current_dt) & (opt["scrip"] == ce_scrip)][
                    "close"
                ].iloc[0],
                opt[(opt["date_time"] == current_dt) & (opt["scrip"] == pe_scrip)][
                    "close"
                ].iloc[0],
            )
            if int(ce_scrip[:-2]) < int(pe_scrip[:-2]) and check_inverted:
                get_straddle_strike(opt, fut, current_dt, end_time, gap, sd=0)
            else:
                return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt

        except Exception as e:
            # print("error occured in get strangle strike", e)
            continue

    return None, None, None, None, None, None

def get_sl_intra_sl(price, sl, intra_sl):
    sl_price = price + (price * sl) / 100
    intra_sl_price = price + (price * intra_sl) / 100
    return sl_price, intra_sl_price

def sre_w_range(option, fut, start_dt, end_dt, sl, intra_sl, om, index, typ, dtes, dte1, dte2, dte3, dte4, dte5, nml_cut):
    sd = 0.0
    if isinstance(om, str):
        if om.lower().endswith("sd"):
            sd = float(om[:-2])
            om = 0.0
        elif om.lower().endswith("s"):
            sd = float(om[:-1])
            om = 0.0
        else:
            om = float(om)
            sd = 0.0
    else:
        om = float(om)
        sd = 0.0
    metadata = {
        "P_Strategy": "SRE_W_Range",
        "P_Index": index.upper(),
        "P_StartTime": start_dt.time(),
        "P_EndTime": end_dt.time(),
        "P_OrderSide": "BUY" if sl > 0 else "SELL",
        "P_sl": sl,
        "P_intraSL": intra_sl,
        "P_OM": om,
        "P_FixedOrDynamic": typ,
        "P_NormalOrCut": nml_cut,
        "Start.Date": start_dt.date(),
        "End.Date": end_dt.date(),
        "dte1": dte1,
        "dte2": dte2,
        "dte3": dte3,
        "dte4": dte4,
        "dte5": dte5,
        "Start.DTE": dte5,
        "End.DTE": dte1,
        "Day.Count": (end_dt.date() - start_dt.date()).days + 1,
    }
    ce_data, pe_data = None, None
    opt = option.copy()
    for i in range(10):
        metadata[f"ST{i}.CE_Scrip"] = None
        metadata[f"ST{i}.PE_Scrip"] = None
        metadata[f"ST{i}.Future"] = None
        metadata[f"ST{i}.Current_dt"] = None
        metadata[f"ST{i}.Price"] = None
        metadata[f"ST{i}.Hit_Time"] = None
        metadata[f"ST{i}.PNL"] = None
        metadata[f"ST{i}.SL_Range"] = None
        metadata[f"ST{i}.Intra_SL_Range"] = None

    count = 0
    while count < 10:
        # print("start_dt", start_dt, "end_dt", end_dt)
        ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt = (get_straddle_strike(opt, fut, start_dt, end_dt, gap=get_gap(opt), sd=sd) if om == 0 else get_strangle_strike(
                opt, fut, start_dt, end_dt, gap=get_gap(opt), om=om, index=index
            )
        )
        if ce_scrip is None or pe_scrip is None:
            # print("No valid straddle/strangle found for the given parameters.")
            return metadata
        metadata[f"ST{count}.Current_dt"] = current_dt
        metadata[f"ST{count}.Future"] = future_price
        metadata[f"ST{count}.CE_Scrip"] = ce_scrip
        metadata[f"ST{count}.PE_Scrip"] = pe_scrip
        # print("ce_scrip, pe_scrip", ce_scrip, pe_scrip)
        scrip = int(ce_scrip[:-2])
        price = ce_price + pe_price
        sl_price, intra_sl_price = get_sl_intra_sl(price, sl, intra_sl)

        sl_range_upper, sl_range_lower = scrip + sl_price, scrip - sl_price
        intra_sl_range_upper, intra_sl_range_lower = (
            scrip + intra_sl_price,
            scrip - intra_sl_price,
        )

        metadata[f"ST{count}.SL_Range"] = (int(sl_range_lower), int(sl_range_upper))
        metadata[f"ST{count}.Intra_SL_Range"] = (
            int(intra_sl_range_lower),
            int(intra_sl_range_upper),
        )

        opt = opt[current_dt:end_dt]
        fut = fut[current_dt:end_dt]
        ce_data = opt[opt["scrip"] == ce_scrip]
        pe_data = opt[opt["scrip"] == pe_scrip]

        # Set high and low equal to close at 9:15
        mask_ce = ce_data.index == pd.Timestamp("09:15:00").time()
        mask_pe = pe_data.index == pd.Timestamp("09:15:00").time()

        if mask_ce.any():
            close_value = ce_data.loc[mask_ce, "close"].values
            ce_data.loc[mask_ce, "high"] = close_value
            ce_data.loc[mask_ce, "low"] = close_value

        if mask_pe.any():
            close_value = pe_data.loc[mask_pe, "close"].values
            pe_data.loc[mask_pe, "high"] = close_value
            pe_data.loc[mask_pe, "low"] = close_value

        common_index = ce_data.index.intersection(pe_data.index)
        ce_data = ce_data.loc[common_index]
        pe_data = pe_data.loc[common_index]
        ce_data = ce_data[ce_data.index.time <= pd.Timestamp("15:15:00").time()]
        pe_data = pe_data[pe_data.index.time <= pd.Timestamp("15:15:00").time()]
        intra_sl_time = None
        sl_time = None
        if typ.lower() == "dynamic":
            unique_dates = ce_data.index.date
            unique_dates = pd.Series(unique_dates).unique()

            for date_idx, current_date in enumerate(unique_dates):
                is_last_day = date_idx == len(unique_dates) - 1
                # Filter data for current date
                date_mask = ce_data.index.date == current_date
                ce_day = ce_data[date_mask]
                pe_day = pe_data[date_mask]
                intra_sl_time_day = ce_day[
                    ((scrip + ce_day["high"] - pe_day["low"]) <= intra_sl_range_lower)
                    | ((scrip + ce_day["low"] - pe_day["high"]) >= intra_sl_range_upper)
                ].index
                sl_time_day = ce_day[
                    ((scrip + ce_day["close"] - pe_day["close"]) <= sl_range_lower)
                    | ((scrip + ce_day["close"] - pe_day["close"]) >= sl_range_upper)
                ].index

                # If SL hit on current day, exit loop
                if len(intra_sl_time_day) > 0 or len(sl_time_day) > 0:
                    intra_sl_time = (
                        intra_sl_time_day[0] if len(intra_sl_time_day) > 0 else None
                    )
                    sl_time = sl_time_day[0] if len(sl_time_day) > 0 else None
                    break

                if not is_last_day:
                    start_dt = ce_data[ce_data.index.date == current_date].index[-1] 
                    ce_scrip, pe_scrip, ce_price, pe_price, future_price,   current_dt = (
                        get_straddle_strike(
                            opt, fut, start_dt, end_dt, gap=get_gap(opt), sd=sd
                        )
                        if om == 0.0
                        else get_strangle_strike(
                            opt,
                            fut,
                            start_dt,
                            end_dt,
                            gap=get_gap(opt),
                            om=om,
                            index=index,
                        ))
                    scrip = int(ce_scrip[:-2])  
                    price = ce_price + pe_price
                    sl_price, intra_sl_price = get_sl_intra_sl(price, sl, intra_sl)

                    sl_range_upper, sl_range_lower = scrip + sl_price, scrip - sl_price
                    intra_sl_range_upper, intra_sl_range_lower = (
                        scrip + intra_sl_price,
                        scrip - intra_sl_price,
                    )            
        else:
            intra_sl_time = ce_data[
                ((scrip + ce_data["high"] - pe_data["low"]) <= intra_sl_range_lower)
                | ((scrip + ce_data["low"] - pe_data["high"]) >= intra_sl_range_upper)
            ].index
            sl_time = ce_data[
                ((scrip + ce_data["close"] - pe_data["close"]) <= sl_range_lower)
                | ((scrip + ce_data["close"] - pe_data["close"]) >= sl_range_upper)
            ].index

            if len(intra_sl_time) > 0:
                intra_sl_time = intra_sl_time[0]
            else:
                intra_sl_time = None
            if len(sl_time) > 0:
                sl_time = sl_time[0]
            else:
                sl_time = None

        # print("intra sl time", intra_sl_time)
        # print("sl time", sl_time)
        pnl = 0
        exit_time = None
        exit_type = None
        if sl_time is None and intra_sl_time is None:
            pnl = (
                price
                - (ce_data["close"].iloc[-1] + pe_data["close"].iloc[-1])
                - price * SLIPAGES[index.lower()]
            )
            exit_time = ce_data.index[-1]
            exit_type = "End of Day"
            metadata[f"ST{count}.PNL"] = pnl
            # print("end of day and time", exit_time)
            return metadata
        if sl_time and intra_sl_time:
            if sl_time < intra_sl_time:
                exit_time = sl_time
                exit_type = "SL Hit"
                metadata[f"ST{count}.Hit_Time"] = exit_time
                # print("sl hit and time", exit_time)
            else:
                exit_time = intra_sl_time
                exit_type = "Intra SL Hit"
                metadata[f"ST{count}.Hit_Time"] = exit_time
                # print("intra sl hit and time", exit_time)

        elif sl_time and not intra_sl_time:
            exit_time = sl_time
            exit_type = "SL Hit"
            metadata[f"ST{count}.Hit_Time"] = exit_time
            # print("only sl hit and time", exit_time)

        elif intra_sl_time and not sl_time:
            exit_time = intra_sl_time
            exit_type = "Intra SL Hit"
            metadata[f"ST{count}.Hit_Time"] = exit_time
            # print("only intra sl hit and time", exit_time)

        if exit_type == "SL Hit":
            pnl = price - sl_price - price * SLIPAGES[index.lower()]
            metadata[f"ST{count}.PNL"] = pnl

        elif exit_type == "Intra SL Hit":
            pnl = (
                price
                - (ce_data.loc[exit_time, "close"] + pe_data.loc[exit_time, "close"])
                - price * SLIPAGES[index.lower()]
            )
            metadata[f"ST{count}.PNL"] = pnl
        count += 1

        if nml_cut.lower() == "cut":

            if exit_time.date() != end_dt.date():
                start_dt = pd.to_datetime(f"{exit_time.date()} 15:15:00")
                # print("..start_dt...", start_dt)
            else:
                return metadata
        else:
            # current_dte = ce_data.loc[exit_time, "dte"]
            current_dte = dtes.loc[exit_time.date(), index.upper()]
            # print("current_dte", current_dte)
            if current_dte == 5 and dte5 > 0:
                dte5 -= 1
                start_dt = exit_time
            elif current_dte == 4 and dte4 > 0:
                dte4 -= 1
                start_dt = exit_time
            elif current_dte == 3 and dte3 > 0:
                dte3 -= 1
                start_dt = exit_time
            elif current_dte == 2 and dte2 > 0:
                dte2 -= 1
                start_dt = exit_time
            elif current_dte == 1 and dte1 > 0:
                dte1 -= 1
                start_dt = exit_time
            else:
                if exit_time.date() != end_dt.date():
                    start_dt = pd.to_datetime(f"{exit_time.date()} 15:15:00")

                    # print(".start_dt...", start_dt)
                else:
                    return metadata

    return metadata

data_path = Path("C:\\PICKLE")
params = pd.read_csv("parameter.csv")
dtes = pd.read_csv("C:\\PICKLE\\DTE.csv", parse_dates=["Date"]).set_index("Date")
dtes.index = pd.to_datetime(dtes.index, format="%d-%m-%Y")
dtes.index = [dt.date() for dt in dtes.index]
start_date = pd.to_datetime(params.loc[0, "start_date"], dayfirst=True)
end_date = pd.to_datetime(params.loc[0, "end_date"], dayfirst=True)
start_times = pd.to_datetime(
    params.start_time.dropna().unique(), format="%H-%M-%S"
).time
end_times = pd.to_datetime(params["end_time"].dropna().unique(), format="%H-%M-%S").time
indexs = params["index"].str.upper().dropna().unique()
sls = params["sl"].dropna().unique()
intra_sls = params["intra_sl"].dropna().unique()
oms = params["om"].dropna().unique()
types = params["static_dynamic"].dropna().unique()
dte1s = params["dte1"].dropna().unique()
dte2s = params["dte2"].dropna().unique()
dte3s = params["dte3"].dropna().unique()
dte4s = params["dte4"].dropna().unique()
dte5s = params["dte5"].dropna().unique()
nml_cuts = params["nml_cut"].dropna().unique()

for index in indexs:

    processed_files = cwd / f"{index}_output_files"
    processed_files.mkdir(exist_ok=True)
    print(f"Processing index: {index}")
    future_path = data_path / f"{PREFIX[index.lower()]} Future"
    options_path = data_path / f"{PREFIX[index.lower()]} Options"

    nifty_options_list, nifty_future_list = load_files_in_date_range(
        options_path, future_path, start_date, end_date
    )
    weekly_groups = group_files_by_week(nifty_future_list, nifty_options_list, index)
    combinations = []
    for typ in types:
        for start_time in start_times:
            for end_time in end_times:
                for sl in sls:
                    for intra_sl in intra_sls:
                        for om in oms:
                            for dte1 in dte1s:
                                for dte2 in dte2s:
                                    for dte3 in dte3s:
                                        for dte4 in dte4s:
                                            for dte5 in dte5s:
                                                for nml_cut in nml_cuts:
                                                    combine = []
                                                    for weekly_group in weekly_groups:
                                                        futs = []
                                                        opts = []
                                                        for (fut_path, opt_path) in weekly_group:
                                                            fut = pd.read_pickle(fut_path)
                                                            opt = pd.read_pickle(opt_path)
                                                            opt["dte"] = dtes.loc[pd.to_datetime(opt["date_time"]).dt.date, index.upper()].values
                                                            futs.append(fut)
                                                            opts.append(opt)

                                                        fut_week = pd.concat(futs).set_index("date_time")
                                                        opt_week = pd.concat(opts).set_index("date_time")
                                                        fut_week.index = pd.to_datetime(fut_week.index)
                                                        opt_week.index = pd.to_datetime(opt_week.index)
                                                        opt_week["date_time"] = opt_week.index
                                                        fut_week["date_time"] = fut_week.index
                                                        entry_file = weekly_group[0][0]
                                                        entry_date = pd.read_pickle(entry_file).iloc[0]["date_time"]
                                                        entry_date = pd.to_datetime(entry_date)
                                                        # Exit: last file's date and exit time
                                                        exit_file = weekly_group[-1][0]
                                                        exit_date = pd.read_pickle(exit_file).iloc[-1]["date_time"]
                                                        exit_date = pd.to_datetime(exit_date)

                                                        print(f"Processing week: {entry_date.date()} to {exit_date.date()}")
                                                        
                                                        start_dt = pd.to_datetime(f"{entry_date.date()} {start_time.strftime('%H:%M:%S')}" )
                                                        end_dt = pd.to_datetime(f"{exit_date.date()} {end_time.strftime('%H:%M:%S')}" )
                                                        
                                                        rows = sre_w_range(
                                                            opt_week,
                                                            fut_week,
                                                            start_dt,
                                                            end_dt,
                                                            sl,
                                                            intra_sl,
                                                            om,
                                                            index,
                                                            typ,
                                                            dtes,
                                                            dte1,
                                                            dte2,
                                                            dte3,
                                                            dte4,
                                                            dte5,
                                                            nml_cut,
                                                        )

                                                        combinations.append(rows)
                                                        combine.append(rows)
                                                    file_path = (
                                                            processed_files
                                                            / f"sre_w_range_{entry_date.strftime('%H%M')}_{exit_date.strftime('%H%M')} {sl} {intra_sl} {om} {int(dte1)} {int(dte2)} {int(dte3)} {int(dte4)} {int(dte5)} {nml_cut}.csv"
                                                        )
                                                    pd.DataFrame(combine).to_csv(
                                                            file_path, index=False
                                                        )

    pd.DataFrame(combinations).to_csv(processed_files / "all_files.csv", index=False)
