import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import gui, tqdm
import os


cwd = Path.cwd()
nifty_future = Path("C:\\PICKLE\\Nifty Future")
nifty_options = Path("C:\\PICKLE\\Nifty Options")
param_df = pd.read_csv("parameter.csv")
start_date = pd.to_datetime(param_df.loc[0, "start_date"], dayfirst=True)
end_date = pd.to_datetime(param_df.loc[0, "end_date"], dayfirst=True)


def get_file_date(file: Path) -> datetime:
    """
    Extract date from filename prefix: YYYY-MM-DD_xxx.pkl
    """
    try:
        date_str = file.stem.split("_")[0]  # take '2019-03-07'
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None


nifty_options_list = []
nifty_future_list = []

options_dict = {}
for file in nifty_options.iterdir():
    if file.is_file():
        file_date = get_file_date(file)
        if file_date and start_date <= file_date <= end_date:
            options_dict[file_date.date()] = file

futures_dict = {}
for file in nifty_future.iterdir():
    if file.is_file():
        file_date = get_file_date(file)
        if file_date and start_date <= file_date <= end_date:
            futures_dict[file_date.date()] = file

# Keep only dates where both exist
common_dates = sorted(set(options_dict.keys()) & set(futures_dict.keys()))
nifty_options_list = [options_dict[d] for d in common_dates]
nifty_future_list = [futures_dict[d] for d in common_dates]


def get_straddle_strike(
    opt: pd.DataFrame,
    fut: pd.DataFrame,
    start_dt: pd.Timedelta,
    end_dt: pd.Timedelta,
    gap: int = 50,
    sd: int = 0,
    SDroundoff: bool = False,
):
    """AI is creating summary for get_straddle_strike

    Args:
        opt (pd.DataFrame): [option data]
        fut (pd.DataFrame): [future data]
        start_dt (pd.Timedelta): [start/entry time]
        end_dt (pd.Timedelta): [end/exit time]
        gap (int, optional): [gap is different between two next scrip in option data]. Defaults to 50.
        sd (int, optional): [description]. Defaults to 0.
        SDroundoff (bool, optional): [description]. Defaults to False.
    """
    valid_times = fut.loc[start_dt:end_dt].index
    # print(valid_times[:2])
    for current_dt in valid_times:
        try:
            future_price = fut.loc[current_dt, "close"]
            # print(future_price, current_dt)
            round_future_price = round(future_price / gap) * gap
            ce_scrip, pe_scrip = f"{round_future_price}CE", f"{round_future_price}PE"
            # print(ce_scrip, pe_scrip)
            ce_price, pe_price = (
                opt[
                    (opt["date_time"] == current_dt) & (opt["scrip"] == ce_scrip)
                ].close.iloc[0],
                opt[
                    (opt["date_time"] == current_dt) & (opt["scrip"] == pe_scrip)
                ].close.iloc[0],
            )
            # print(ce_price, pe_price)
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
            # print(ce_scrip_list, pe_scrip_list)
            scrip_index, min_value = None, float("inf")
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
            if sd:
                sd_range = (ce_price + pe_price) * sd

                if SDroundoff:
                    sd_range = round(sd_range / gap) * gap
                else:
                    sd_range = max(gap, round(sd_range / gap) * gap)

            ce_scrip, pe_scrip = (
                f"{int(ce_scrip[:-2])+sd_range}CE",
                f"{int(pe_scrip[:-2])}PE",
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
    opt: pd.DataFrame,
    fut: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    gap: int = 50,
    om: float = None,
):
    """
    Args:
        opt (pd.DataFrame): [description]
        fut (pd.DataFrame): [description]
        start_dt (pd.Timedelta): [description]
        end_dt (pd.Timedelta): [description]
        gap (int, optional): [description]. Defaults to 50.
        om (float, optional): [description]. Defaults to None.
    """
    valid_times = fut.loc[start_dt:end_dt].index
    # lat's take example
    for current_dt in valid_times:
        try:
            future_price = fut.loc[current_dt, "close"]
            # future price = 17510.4 at current date and close
            one_om = get_one_om(fut, future_price)
            # one om = 170.0
            target = one_om * om
            # target = 170.0 * 0.4  lats take om=0.4
            # target = 68.0

            # keep data only with index current dt and ehose close is greater target
            target_od = (
                opt[(opt.index == current_dt) & (opt["close"] >= target)]
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
                ce_scrip,
                ce_scrip,
                f"{int(ce_scrip[:-2])-gap}CE",
                f"{int(ce_scrip[:-2])+gap}CE",
            ]
            pe_scrip_list = [
                pe_scrip,
                f"{int(pe_scrip[:-2])-gap}PE",
                f"{int(pe_scrip[:-2])+gap}PE",
                pe_scrip,
                pe_scrip,
            ]

            # ce_scrip_list ['17550CE', '17550CE', '17550CE', '17500CE', '17600CE']
            # pe_scrip_list ['17400PE', '17350PE', '17450PE', '17400PE', '17400PE']
            # take each pair and compair min difference of price of scrip

            ce_price_list, pe_price_list = [], []
            for i in range(5):
                try:
                    ce_price_list.append(
                        opt[
                            (opt["date_time"] == current_dt)
                            & (opt["scrip"] == ce_scrip_list[i])
                        ]["close"].iloc[0]
                    )
                except:
                    ce_scrip_list.append(0)
                    print("price not found")
                try:
                    pe_price_list.append(
                        opt[
                            (opt["date_time"] == current_dt)
                            & (opt["scrip"] == pe_scrip_list[i])
                        ]["close"].iloc[0]
                    )
                except:
                    pe_price_list.append(0)
                    print("price not found")
            # print(ce_price_list, pe_price_list)
            # ce price list contain prices at scrip
            # ce_price_list [np.float64(68.3), np.float64(68.3), np.float64(68.3), np.float64(92.25), np.float64(49.4)]
            # pe_price_list [np.float64(79.65), np.float64(64.2), np.float64(99.1), np.float64(79.65), np.float64(79.65)]
            target_2, target_3 = target * 2, target * 3
            # target_2 = 136.0 =  68 * 2
            # target_3 =204.0 =  68 * 3

            min_diff = float("inf")
            scrip_index = None

            for i in range(5):
                if (
                    (min_diff > abs(ce_price_list[i] - pe_price_list[i]))
                    and (ce_price_list[i] + pe_price_list[i] >= target_2)
                    and (ce_price_list[i] + pe_price_list[i] <= target_3)
                ):
                    min_diff = abs(ce_price_list[i] - pe_price_list[i])
                    scrip_index = i

            # min_diff = 11.35

            ce_scrip, pe_scrip = ce_scrip_list[scrip_index], pe_scrip_list[scrip_index]
            # ce_scrip 17550CE
            # pe_scrip 17400PE
            ce_price, pe_price = (
                opt[(opt["date_time"] == current_dt) & (opt["scrip"] == ce_scrip)][
                    "close"
                ].iloc[0],
                opt[(opt["date_time"] == current_dt) & (opt["scrip"] == pe_scrip)][
                    "close"
                ].iloc[0],
            )
            # ce_price 68.3
            # pe_price 79.65
            # print(ce_scrip,pe_scrip)

            # return this  ('17550CE', '17400PE', 68.3, 79.65, 17510.4, '2022-01-03 09:20:00')
            return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt

        except Exception as e:
            print(f"error occured in get strangle {e}")
            continue
    return None, None, None, None, None, None  # else return None

def ut_check(data, sl_time, ut_sl):
    price = data.loc[sl_time, "close"]
    ut_sl_price = price + ((price * ut_sl) / 100)
    data = data[data.index > sl_time]

    try:
        ut_sl_time = data[data["high"] >= ut_sl_price].index[0]
    except:
        ut_sl_time = None

    if ut_sl_time:
        return ut_sl_price, price, ut_sl_time
    else:
        return data["close"].iloc[-1], price, data.index[-1]

def b120(
    ce_data,
    pe_data,
    ce_price,
    pe_price,
    ce_sl_price,
    pe_sl_price,
    future_price,
    ce_scrip,
    pe_scrip,
    sl,
    ut_sl,
    om,
    current_dt,
    exit_time,
):
    method = "HL"
    meta_data = {
        "P_Strategy": "B120",
        "P_Index": "NIFTY",
        "P_StartTime": current_dt.time(),
        "P_EndTime": exit_time.time(),
        "P_OrderSide": "SELL",
        "P_Method": method,
        "P_SL": sl,
        "P_UTSL": ut_sl,
        "P_OM": om,
        "Date": current_dt.date(),
        "Day": current_dt.day_name(),
        "DTE": ce_data["dte"].iloc[0] + 1,
        "EntryTime": current_dt.time(),
        "Future": future_price,
        "CE.Strike": ce_data["scrip"].iloc[0],
        "CE.Open": ce_price,
        "CE.High": ce_data["high"].max(),
        "CE.Low": ce_data["low"].min(),
        "CE.Close": ce_data["close"].iloc[-1],
        "CE.SL.Flag": False,  # update in sl time compairison
        "CE.SL.Time": None,
        "CE.PNL": None,
        "PE.Strike": pe_data["scrip"].iloc[0],
        "PE.Open": pe_price,
        "PE.High": pe_data["high"].max(),
        "PE.Low": pe_data["low"].min(),
        "PE.Close": pe_data["close"].iloc[-1],
        "PE.SL.Flag": False,
        "PE.SL.Time": None,
        "PE.PNL": None,
        "UT.Strike": None,
        "UT.Open": None,
        "UT.High": None,
        "UT.Low": None,
        "UT.Close": None,
        "UT.SL.Flag": False,
        "UT.SL.Time": None,
        "BPL": None,
        "TT.PL.AT.SL": 0,
        "UT.PL.AT.SL": 0,
        "UT.PNL": 0,
        "Total.PNL": None,
    }
    
    try:
        ce_sl_time = ce_data[ce_data["high"] >= ce_sl_price].index[0]
    except:
        ce_sl_time = None
    try:
        pe_sl_time = pe_data[pe_data["high"] >= pe_sl_price].index[0]
    except:
        pe_sl_time = None
    ce_pnl, pe_pnl = None, None
    
    if ce_sl_time and pe_sl_time:
        if ce_sl_time < pe_sl_time:
            # print(f"ce sl hit first at {ce_sl_time}")
            meta_data["CE.SL.Flag"] = True
            meta_data["CE.SL.Time"] = ce_sl_time.time()
            meta_data["PE.SL.Flag"] = True
            meta_data["PE.SL.Time"] = pe_sl_time.time()
            close_price, pe_ut_price, sl_time = ut_check(pe_data.copy(), ce_sl_time, ut_sl)
            ut_pnl = pe_ut_price - close_price
            pe_pnl = pe_price - pe_sl_price - (pe_price * 0.01)
            ce_pnl = ce_price - ce_sl_price - (ce_price * 0.01)
            meta_data["CE.PNL"] = ce_pnl
            meta_data["PE.PNL"] = pe_pnl
            meta_data["UT.PNL"] = ut_pnl
            meta_data["UT.Strike"] = pe_data["scrip"].iloc[0]
            meta_data["UT.Open"] = pe_data.loc[ce_sl_time, "close"]
            meta_data["UT.High"] = pe_data.loc[ce_sl_time:exit_time, "high"].max()
            meta_data["UT.Low"] = pe_data.loc[ce_sl_time:exit_time, "low"].min()
            meta_data["UT.Close"] = pe_data.loc[exit_time, "close"]
            meta_data["UT.SL.Flag"] = True if sl_time != pe_data.index[-1] else False
            meta_data["UT.SL.Time"] = (
                sl_time.time() if sl_time != pe_data.index[-1] else None
            )
            meta_data["BPL"] = 0
            meta_data["TT.PL.AT.SL"] = ce_pnl
            meta_data["UT.PL.AT.SL"] = (
                pe_price - pe_data.loc[ce_sl_time, "close"] - (pe_price * 0.01)
            )
            meta_data["Total.PNL"] = (
                meta_data["TT.PL.AT.SL"] + meta_data["UT.PL.AT.SL"] + ut_pnl
            )
            # print("pe price",pe_price,"pe at st time",pe_data.loc[ce_sl_time,'close'])

            # print("ce pnl", ce_pnl, "ce pnl", pe_pnl, "ut pnl", ut_pnl, "UT.PL.AT.SL",meta_data["UT.PL.AT.SL"])
        else:
            # print(f"pe sl hit first at {pe_sl_time}")
            meta_data["CE.SL.Flag"] = False
            meta_data["CE.SL.Time"] = ce_sl_time.time()
            meta_data["PE.SL.Flag"] = True
            meta_data["PE.SL.Time"] = pe_sl_time.time()
            close_price, ce_ut_price, sl_time = ut_check(ce_data.copy(), pe_sl_time, ut_sl)
            ut_pnl = ce_ut_price - close_price
            ce_pnl = ce_price - ce_sl_price - (ce_price * 0.01)
            pe_pnl = pe_price - pe_sl_price - (pe_price * 0.01)
            meta_data["CE.PNL"] = ce_pnl
            meta_data["PE.PNL"] = pe_pnl
            meta_data["UT.PNL"] = ut_pnl
            meta_data["UT.Strike"] = ce_data["scrip"].iloc[0]
            meta_data["UT.Open"] = ce_data.loc[pe_sl_time, "close"]
            meta_data["UT.High"] = ce_data.loc[pe_sl_time:exit_time, "high"].max()
            meta_data["UT.Low"] = ce_data.loc[pe_sl_time:exit_time, "low"].min()
            meta_data["UT.Close"] = ce_data.loc[exit_time, "close"]
            meta_data["UT.SL.Flag"] = True if sl_time != ce_data.index[-1] else False
            meta_data["UT.SL.Time"] = (
                sl_time.time() if sl_time != ce_data.index[-1] else None
            )
            meta_data["BPL"] = 0
            meta_data["TT.PL.AT.SL"] = pe_pnl
            meta_data["UT.PL.AT.SL"] = (
                ce_price - ce_data.loc[pe_sl_time, "close"] - (ce_price * 0.01)
            )
            meta_data["Total.PNL"] = pe_pnl + meta_data["UT.PL.AT.SL"] + ut_pnl
            # print("ce sl hit at",sl_time)
            # print("pe pnl",pe_pnl,"ce pnl",ce_pnl,"ut pnl",ut_pnl)
    elif ce_sl_time is None and pe_sl_time is None:
        ce_pnl = ce_price - ce_data.loc[exit_time, "close"] - (ce_price * 0.01)
        meta_data["CE.PNL"] = ce_pnl
        meta_data["BPL"] = ce_pnl + (
            pe_price - pe_data.loc[exit_time, "close"] - (pe_price * 0.01)
        )
        meta_data["PE.PNL"] = (
            pe_price - pe_data.loc[exit_time, "close"] - (pe_price * 0.01)
        )
        meta_data["Total.PNL"] = meta_data["BPL"]
        # print("ce sl and pe sl not hit","pnl",ce_pnl,meta_data["PE.PNL"])
    elif pe_sl_time is None and ce_sl_time is not None:
        meta_data["CE.SL.Flag"] = True
        meta_data["CE.SL.Time"] = ce_sl_time.time()
        data = pe_data[ce_sl_time:]
        close_price, pe_ut_price, sl_time = ut_check(data, ce_sl_time, ut_sl)
        ut_pnl = pe_ut_price - close_price
        pe_pnl = pe_price - pe_data.loc[exit_time, "close"] - (pe_price * 0.01)
        ce_pnl = ce_price - ce_sl_price - (ce_price * 0.01)
        meta_data["CE.PNL"] = ce_pnl
        meta_data["PE.PNL"] = pe_pnl
        meta_data["UT.PNL"] = ut_pnl
        meta_data["UT.Strike"] = pe_scrip

        meta_data["UT.Open"] = pe_data.loc[ce_sl_time, "close"]
        meta_data["UT.High"] = pe_data.loc[ce_sl_time:exit_time, "high"].max()
        meta_data["UT.Low"] = pe_data.loc[ce_sl_time:exit_time, "low"].min()
        meta_data["UT.Close"] = pe_data.loc[exit_time, "close"]
        meta_data["UT.SL.Flag"] = True if sl_time != pe_data.index[-1] else False
        meta_data["UT.SL.Time"] = (
            sl_time.time() if sl_time != pe_data.index[-1] else None
        )
        meta_data["BPL"] = 0
        meta_data["TT.PL.AT.SL"] = ce_pnl
        meta_data["UT.PL.AT.SL"] = (
            pe_price - pe_data.loc[ce_sl_time, "close"] - (pe_price * 0.01)
        )
        meta_data["Total.PNL"] = (
            meta_data["TT.PL.AT.SL"] + meta_data["UT.PL.AT.SL"] + ut_pnl
        )
    elif pe_sl_time is not None and ce_sl_time is None:
        meta_data["PE.SL.Flag"] = True
        meta_data["PE.SL.Time"] = pe_sl_time.time()
        data = ce_data[pe_sl_time:]
        # print(ce_sl_time,pe_sl_time)
        close_price, ce_ut_price, sl_time = ut_check(data, pe_sl_time, ut_sl)
        # print("after ut check")
        ut_pnl = ce_ut_price - close_price
        ce_pnl = ce_price - ce_data.loc[exit_time, "close"] - (ce_price * 0.01)
        pe_pnl = pe_price - pe_sl_price - (pe_price * 0.01)
        meta_data["CE.PNL"] = ce_pnl
        meta_data["PE.PNL"] = pe_pnl
        meta_data["UT.PNL"] = ut_pnl
        meta_data["UT.Strike"] = ce_scrip
        meta_data["UT.Open"] = ce_data.loc[pe_sl_time, "close"]
        meta_data["UT.High"] = ce_data.loc[pe_sl_time:exit_time, "high"].max()
        meta_data["UT.Low"] = ce_data.loc[pe_sl_time:exit_time, "low"].min()
        meta_data["UT.Close"] = ce_data.loc[exit_time, "close"]
        meta_data["UT.SL.Flag"] = True if sl_time != ce_data.index[-1] else False
        meta_data["UT.SL.Time"] = (
            sl_time.time() if sl_time != ce_data.index[-1] else None
        )
        meta_data["BPL"] = 0
        meta_data["TT.PL.AT.SL"] = pe_pnl
        meta_data["UT.PL.AT.SL"] = (
            ce_price - ce_data.loc[pe_sl_time, "close"] - (ce_price * 0.01)
        )
        meta_data["Total.PNL"] = pe_pnl + meta_data["UT.PL.AT.SL"] + ut_pnl

    return meta_data

def b120_intraday(
    option, future, start_time, end_time, method, sl, ut_sl, om, trade_date
):
    if om != 0:
        ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt = (
            get_strangle_strike(option, future, start_time, end_time, 50, om)
        )
    else:
        ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt = (
            get_straddle_strike(option, future, start_time, end_time)
        )

    # Calculate CE and PE stop loss prices as sl percentage above their entry prices
    ce_sl_price = ce_price + ((ce_price * sl) / 100)
    pe_sl_price = pe_price + ((pe_price * sl) / 100)
    option = option.loc[start_time:end_time]

    option = option = option[
        option.index > start_time
    ]  # to avoid entry time data in sl hit

    ce_data = option[option["scrip"] == ce_scrip]
    pe_data = option[option["scrip"] == pe_scrip]

    # print("ce start", ce_data.index[0], "pe start", pe_data.index[0])
    data = b120(
        ce_data,
        pe_data,
        ce_price,
        pe_price,
        ce_sl_price,
        pe_sl_price,
        future_price,
        ce_scrip,
        pe_scrip,
        sl,
        ut_sl,
        om,
        current_dt,
        end_time,
    )

    return data

start_times = pd.to_datetime(
    param_df.start_time.dropna().unique(), format="%H:%M:%S"
).time
end_times = pd.to_datetime(param_df.end_time.dropna().unique(), format="%H:%M:%S").time
sls = param_df.sl.dropna().unique()
ut_sls = param_df.ut_sl.dropna().unique()
oms = param_df.om.dropna().unique()
methods = param_df.method.dropna().unique()


# ************** run for all files ****************

for future_path, option_path in zip(nifty_future_list, nifty_options_list):
    option = pd.read_pickle(option_path).set_index("date_time")
    fut = pd.read_pickle(future_path).set_index("date_time")
    option.index = pd.to_datetime(option.index)
    fut.index = pd.to_datetime(fut.index)
    option["date_time"] = option.index
    fut["date_time"] = fut.index

    file_date = fut.index[0].date()

    combinations = []
    for start_time in tqdm(start_times, colour="red"):
        for end_time in end_times:
            for method in methods:
                for sl in sls:
                    for ut_sl in ut_sls:
                        for om in oms:
                            # print(start_date,end_time,method,sl,om)
                            start_dt = pd.to_datetime(
                                f"{file_date} {start_time.strftime('%H:%M:%S')}"
                            )
                            end_dt = pd.to_datetime(
                                f"{file_date} {end_time .strftime('%H:%M:%S')}"
                            )
                            rows = b120_intraday(
                                option.copy(),
                                fut.copy(),
                                start_time=start_dt,
                                end_time=end_dt,
                                method=method,
                                sl=sl,
                                ut_sl=ut_sl,
                                om=om,
                                trade_date=option.index[0].date(),
                            )
                            combinations.append(rows)

    # combination = pd.DataFrame(combinations)
    os.makedirs(cwd / "B120 output", exist_ok=True)
    output_path = cwd / "B120 output"
    pd.DataFrame(combinations).to_csv(
        output_path / f"NIFTY {fut.index.date[0]} B120_all.csv", index=False
    )
