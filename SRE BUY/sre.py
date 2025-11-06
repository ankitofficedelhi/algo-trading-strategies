import pandas as pd
from datetime import datetime ,time
from pathlib import Path
from tqdm import gui, tqdm
import os
import numpy as np

cwd = Path.cwd()
param_df = pd.read_csv("parameter.csv")
param_df["start_date"] = pd.to_datetime(param_df["start_date"], dayfirst=True)
param_df["end_date"] = pd.to_datetime(param_df["end_date"], dayfirst=True)
param_df["start_time"] = pd.to_datetime(param_df["start_time"], format="%H-%M-%S").dt.time
param_df["end_time"] = pd.to_datetime(param_df["end_time"], format="%H-%M-%S").dt.time

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
):
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
    start_datetime = pd.Timestamp.combine(current_date, start_dt) if isinstance(start_dt, time) else start_dt
    end_datetime = pd.Timestamp.combine(current_date, end_dt) - pd.Timedelta(minutes=10) if isinstance(end_dt, time) else end_dt - pd.Timedelta(minutes=10)
    
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
            if sd !=0:
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
    end_datetime = pd.Timestamp.combine(current_date, end_time) - pd.Timedelta(minutes=10)
    
    valid_times = fut.loc[start_datetime:end_datetime].index
    for current_dt in valid_times:
        try:
            future_price = fut.loc[current_dt, "close"]
            # future price = 17510.4
            one_om = get_one_om(fut, future_price, STEP=STEPS[index])
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
                get_straddle_strike(opt, fut, current_dt, end_time, gap,sd)
            else:
                return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt

        except Exception as e:
            # print("error occured in get strangle strike", e)
            continue

    return None, None, None, None, None, None

def check_sl_hit(ce_data, pe_data, sl_price, intra_sl_price):
    ce_data = ce_data[ce_data.index > ce_data.index[0]]
    pe_data = pe_data[pe_data.index > pe_data.index[0]]
    
    try:
        sl_time = ce_data[ce_data.close + pe_data.close <= sl_price].index[0]
    except:
        sl_time = None
    try:
        intra_sl_time = ce_data[
            np.minimum(ce_data.high + pe_data.low, ce_data.low + pe_data.high)
            <= intra_sl_price
        ].index[0]

    except:
        intra_sl_time = None

    return sl_time, intra_sl_time

def calculate_sl_intra(entry_time, ce_data, pe_data, sl, intra_sl):
    ce_pe_price = ce_data.loc[entry_time, "close"] + pe_data.loc[entry_time, "close"]
    sl_price = ce_pe_price * (1 - sl / 100)
    intra_sl_price = ce_pe_price * (1 - intra_sl / 100)
    return sl_price, ce_pe_price, intra_sl_price

def sre(index, start_date, end_date, start_time, end_time, om, sd, sl, intra_sl):

    file_path = Path("C:\\PICKLE\\")
    opt_file_path = file_path / f"{PREFIX[index]} Options"
    fut_file_path = file_path / f"{PREFIX[index]} Future"

    opt_file_list, fut_file_list = load_files_in_date_range(
        opt_file_path, fut_file_path, start_date, end_date
    )
    meta = {
        "P_Strategy": "SRE",
        "P_Index": index,
        "P_StartTime": start_time,
        "P_EndTime": end_time,
        "P_OrderSide": "BUY",
        "P_SL": sl,
        "P_IntraSL": intra_sl,
        "P_OM": om,
    }
    combinations = []
    for opt_file, fut_file in zip(opt_file_list, fut_file_list):
        fut = pd.read_pickle(fut_file).set_index("date_time")
        opt = pd.read_pickle(opt_file).set_index("date_time")
        fut.index = pd.to_datetime(fut.index)
        opt.index = pd.to_datetime(opt.index)
        fut["date_time"] = fut.index
        opt["date_time"] = opt.index

        metadata = meta.copy()
        metadata["Date"] = opt.index[0].date()
        metadata["day"] = pd.to_datetime(opt.index[0].date()).day_name()
        metadata["dte"] = opt.iloc[0].dte + 1

        gap = get_gap(opt)
        ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt = (
            get_strangle_strike(opt, fut, start_time, end_time, gap, om, index=index)
            if om != 0
            else get_straddle_strike(
                opt,
                fut,
                start_time,
                end_time,
                gap,
                sd
            )
        )
        if ce_scrip is None or pe_scrip is None:
            print(
                f"Straddle/Strangle not found for {opt_file.stem} and {fut_file.stem}"
            )
            continue
        print("Selected Strikes:", ce_scrip, pe_scrip, "at", current_dt)
        metadata["EntryTime"] = current_dt.time()
        metadata["Future"] = future_price
        metadata["CE.Price"] = ce_price
        metadata["PE.Price"] = pe_price

        # Convert end_time to datetime by combining with current date
        end_datetime = pd.Timestamp.combine(current_dt.date(), end_time)
        opt = opt.loc[current_dt:end_datetime]
        ce_data = opt[(opt["scrip"] == ce_scrip)]
        pe_data = opt[(opt["scrip"] == pe_scrip)]

        # Extract only common index data
        common_index = ce_data.index.intersection(pe_data.index)
        ce_data = ce_data.loc[common_index]
        pe_data = pe_data.loc[common_index]

        # print("ce data",ce_data.head())
        # print("-----")

        # print("pe data",pe_data.head())
        # print("-----")
        # print("ce close")
        
        entry_time = current_dt
        print("processing...", entry_time)
        sl_price, ce_pe_price, intra_sl_price = calculate_sl_intra(
            entry_time, ce_data, pe_data, sl, intra_sl
        )
        metadata["SL"] = sl_price
        metadata["IntraSL"] = intra_sl_price

        sl_time, intra_sl_time = check_sl_hit(
            ce_data, pe_data, sl_price, intra_sl_price
        )
        print("SL Time:", sl_time, "Intra SL Time:", intra_sl_time)
        sl_count = 0
        total_pnl = 0

        for i in range(8):
            metadata[f"STD{i}.Scrip"] = ""
            metadata[f"STD{i}.Open"] = ""
            metadata[f"STD{i}.SL.Flag"] = False
            metadata[f"STD{i}.SL.Time"] = ""
            metadata[f"STD{i}.INTRA_SL.Flag"] = False
            metadata[f"STD{i}.INTRA_SL.Time"] = ""
            metadata[f"STD{i}.PNL"] = 0
            
        metadata[f"STD0.Scrip"] = (ce_scrip, pe_scrip)
        metadata[f"STD0.Open"] = ce_pe_price
        if sl == 0:
            exit_ce_pe = (
                ce_data.iloc[-1]["close"] + pe_data.iloc[-1]["close"]
            )
            pnl = exit_ce_pe - ce_pe_price - SLIPAGES[index] * ce_pe_price
            metadata["STD0.PNL"] = pnl
            total_pnl += pnl
            combinations.append(metadata)
            break
        while sl_count < 8:

            if sl_time is None and intra_sl_time is None:
                print("1 EXIT TIME HIT ")
                if ce_data.iloc[1:].empty or pe_data.iloc[1:].empty:
                    print("No data available for exit calculation.")
                    break
                
                exit_ce_pe = (
                    ce_data.iloc[-1]["close"] + pe_data.iloc[-1]["close"]
                )
                pnl = exit_ce_pe - ce_pe_price - SLIPAGES[index] * ce_pe_price
                total_pnl += pnl
                metadata[f"STD{sl_count}.PNL"] = round(pnl, 2)
                print(
                    f"Breaking loop - sl_count: {sl_count}, total_pnl: {total_pnl}"
                )  # Debug line
                break
            elif sl_time is not None and intra_sl_time is not None:
                if sl_time < intra_sl_time:
                    # print("2.5 SL HIT ", sl_price)
                    price = ce_data.loc[sl_time, "close"] + pe_data.loc[sl_time, "close"]
                    pnl = price - ce_pe_price - SLIPAGES[index] * ce_pe_price
                    total_pnl += pnl
                    entry_time = sl_time
                    metadata[f"STD{sl_count}.SL.Flag"] = True
                    metadata[f"STD{sl_count}.SL.Time"] = str(sl_time)
                    metadata[f"STD{sl_count}.PNL"] = round(pnl,2)
                    sl_count += 1
                else:
                    pnl = intra_sl_price - ce_pe_price - SLIPAGES[index] * ce_pe_price
                    # print("3",pnl)
                    total_pnl += pnl
                    entry_time = intra_sl_time
                    metadata[f"STD{sl_count}.INTRA_SL.Flag"] = True
                    metadata[f"STD{sl_count}.INTRA_SL.Time"] = str(intra_sl_time)
                    metadata[f"STD{sl_count}.PNL"] = round(pnl, 2)
                    sl_count += 1
            elif sl_time is not None and intra_sl_time is None:
                # print("4 SL HIT only ", sl_price)
                price = ce_data.loc[sl_time, "close"] + pe_data.loc[sl_time, "close"]
                pnl = price - ce_pe_price - SLIPAGES[index] * ce_pe_price
                total_pnl += pnl
                entry_time = sl_time
                metadata[f"STD{sl_count}.SL.Flag"] = True
                metadata[f"STD{sl_count}.SL.Time"] = str(sl_time)
                metadata[f"STD{sl_count}.PNL"] = round(pnl, 2)
                sl_count += 1     
            else:
                # print("5 INTRA SL HIT ", intra_sl)
                pnl = intra_sl_price - ce_pe_price - SLIPAGES[index] * ce_pe_price
                # print("5",pnl)
                total_pnl += pnl
                entry_time = intra_sl_time
                metadata[f"STD{sl_count}.INTRA_SL.Flag"] = True
                metadata[f"STD{sl_count}.INTRA_SL.Time"] = str(intra_sl_time)
                metadata[f"STD{sl_count}.PNL"] = round(pnl, 2)
                sl_count += 1

            if sl_time is not None or intra_sl_time is not None:
                # Compute next re-entry strikes starting from the latest hit time
                ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt = (
                    get_strangle_strike(
                        opt, fut, entry_time.time(), end_time, gap, om, index=index
                    )
                    if om != 0
                    else get_straddle_strike(
                        opt,
                        fut,
                        entry_time,
                        end_time,
                        gap,
                        sd
                    )
                )
                if ce_scrip is None or pe_scrip is None:
                    print(
                            f"Straddle/Strangle not found for {opt_file.stem} and {fut_file.stem}"
                        )
                    break

                entry_time = current_dt

                opt_window = opt.loc[entry_time:end_datetime]
                ce_data = opt_window[(opt_window["scrip"] == ce_scrip)]
                pe_data = opt_window[(opt_window["scrip"] == pe_scrip)]

                common_index = ce_data.index.intersection(pe_data.index)
                ce_data = ce_data.loc[common_index]
                pe_data = pe_data.loc[common_index]

                sl_price, ce_pe_price, intra_sl_price = calculate_sl_intra(
                    entry_time, ce_data, pe_data, sl, intra_sl
                )

                metadata[f"STD{sl_count}.Scrip"] = (ce_scrip, pe_scrip)
                metadata[f"STD{sl_count}.Open"] = ce_pe_price

                sl_time, intra_sl_time = check_sl_hit(
                    ce_data,
                    pe_data,
                    sl_price,
                    intra_sl_price,
                )

        combinations.append(metadata)
    return total_pnl, combinations


processed_files = cwd / "output"
processed_files.mkdir(exist_ok=True)

for idx, rows in param_df.iterrows():
    index = rows["index"].lower()
    start_date = rows["start_date"]
    end_date = rows["end_date"]
    start_time = rows["start_time"]
    end_time = rows["end_time"]
    sl = rows["sl"]
    intra_sl = rows["intra_sl"]
    om = rows["om"]
    sd = 0.0
    if isinstance(om, str):
        if om.lower().endswith('sd'):
            sd = float(om[:-2])
            om = 0.0
        elif om.lower().endswith('s'):
            sd = float(om[:-1])
            om = 0.0
        else:
            om = float(om)
            sd = 0.0
    else:
        om = float(om)
        sd = 0.0
    
    print(om,sd)

    if sl > intra_sl:
        print("SL cannot be greater than Intra SL")
        continue

    total_pnl, combinat = sre(
        index, start_date, end_date, start_time, end_time, om, sd, sl, intra_sl
    )
    pd.DataFrame(combinat).to_csv(
        f"{processed_files}\\SRE {index.upper()} {start_time.strftime('%H%M')} {end_time.strftime('%H%M')} {sl} {intra_sl} {om}.csv", index=False
    )
    print(
        f"SRE {index.upper()} {start_time.strftime('%H%M')} {end_time.strftime('%H%M')} SL: {sl} Intra SL: {intra_sl} OM: {om} Total PnL: {total_pnl} competed successfully."
    )
    