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

# option, future, start_time, end_time, 50, om
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
):

    valid_times = fut.loc[start_time:end_time].index
    for current_dt in valid_times:
        try:
            future_price = fut.loc[current_dt, "close"]
            # future price = 17510.4
            one_om = get_one_om(fut, future_price)
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
                    print(
                        "call list price is empty and my date time column is not present in your data"
                    )
                try:
                    put_list_prices.append(
                        opt[
                            (opt["date_time"] == current_dt)
                            & (opt["scrip"] == pe_scrip_list[z])
                        ]["close"].iloc[0]
                    )
                except:
                    put_list_prices.append(0)
                    print(
                        "put list price is empty and my date time column is not present in your data"
                    )

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
                get_straddle_strike(fut, current_dt, end_time, opt)
            else:
                return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt

        except Exception as e:
            print("error occured in get strangle strike", e)
            continue

    return None, None, None, None, None, None


def ut_check(data, sl_time, ut_sl):
    data = data.loc[sl_time:]
    price = data.iloc[0]["close"]
    # price = data.loc[sl_time, "close"]
    if ut_sl == 0:
        return data["close"].iloc[-1], price, data.index[-1]

    ut_sl_price = price + ((price * ut_sl) / 100)
    data = data[data.index > data.index[0]]

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
    idx
):
    meta_data = {
        "Date": current_dt.date(),
        f"P_StartTime_{idx}": current_dt.time(),
        f"P_EndTime_{idx}": exit_time.time(),
        f"P_SL_{idx}": sl,
        f"P_UTSL_{idx}": ut_sl,
        f"P_OM_{idx}": om,
        f"Future_{idx}": future_price,
        f"CE.Strike_{idx}": ce_data["scrip"].iloc[0],
        f"CE.Price_{idx}": ce_price,
        f"CE.SL.Time_{idx}": None,
        f"CE.PNL_{idx}": None,
        f"PE.Strike_{idx}": pe_data["scrip"].iloc[0],
        f"PE.Price_{idx}": pe_price,
        f"PE.SL.Time_{idx}": None,
        f"PE.PNL_{idx}": None,
        f"UT.Strike_{idx}": None,
        f"UT.Price_{idx}": None,
        f"UT.SL.Time_{idx}": None,
        f"BPL_{idx}": None,
        f"TT.PL.AT.SL_{idx}": 0,
        f"UT.PL.AT.SL_{idx}": 0,
        f"UT.PNL_{idx}": 0,
        f"Total.PNL_{idx}": None,
    }
    ce_sl_time, pe_sl_time = None, None

    # if ce_price != ce_sl_price and pe_price!=pe_sl_price:
    if sl == 0 :
        ce_sl_time,pe_sl_time = None,None
    else:
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
            meta_data[f"CE.SL.Time_{idx}"] = ce_sl_time.time()
            meta_data[f"PE.SL.Time_{idx}"] = pe_sl_time.time()
            close_price, pe_ut_price, sl_time = ut_check(
                pe_data.copy(), ce_sl_time, ut_sl
            )
            ut_pnl = pe_ut_price - close_price
            pe_pnl = pe_price - pe_sl_price - (pe_price * 0.01)
            ce_pnl = ce_price - ce_sl_price - (ce_price * 0.01)
            meta_data[f"CE.PNL_{idx}"] = round(ce_pnl, 2)
            meta_data[f"PE.PNL_{idx}"] = round(pe_pnl, 2)
            meta_data[f"UT.PNL_{idx}"] = round(ut_pnl, 2)
            meta_data[f"UT.Strike_{idx}"] = pe_data["scrip"].iloc[0]
            meta_data[f"UT.Price_{idx}"] = pe_data.loc[ce_sl_time:, "close"].iloc[0]
            meta_data[f"UT.SL.Time_{idx}"] = (
                sl_time.time() if sl_time != pe_data.index[-1] else None
            )
            meta_data[f"BPL_{idx}"] = 0
            meta_data[f"TT.PL.AT.SL_{idx}"] = round(ce_pnl, 2)
            meta_data[f"UT.PL.AT.SL_{idx}"] = (
                pe_price - pe_data.loc[ce_sl_time:, "close"].iloc[0] - (pe_price * 0.01)
            )
            meta_data[f"Total.PNL_{idx}"] = (
                meta_data[f"TT.PL.AT.SL_{idx}"] + meta_data[f"UT.PL.AT.SL_{idx}"] + ut_pnl
            )
            # print("pe price",pe_price,"pe at st time",pe_data.loc[ce_sl_time,'close'])

            # print("ce pnl", ce_pnl, "ce pnl", pe_pnl, "ut pnl", ut_pnl, "UT.PL.AT.SL",meta_data["UT.PL.AT.SL"])
        else:
            # print(f"pe sl hit first at {pe_sl_time}")
            meta_data[f"CE.SL.Time_{idx}"] = ce_sl_time.time()
            meta_data[f"PE.SL.Time_{idx}"] = pe_sl_time.time()
            close_price, ce_ut_price, sl_time = ut_check(
                ce_data.copy(), pe_sl_time, ut_sl
            )
            ut_pnl = ce_ut_price - close_price
            ce_pnl = ce_price - ce_sl_price - (ce_price * 0.01)
            pe_pnl = pe_price - pe_sl_price - (pe_price * 0.01)
            meta_data[f"CE.PNL_{idx}"] = round(ce_pnl, 2)
            meta_data[f"PE.PNL_{idx}"] = round(pe_pnl, 2)
            meta_data[f"UT.PNL_{idx}"] = round(ut_pnl, 2)
            meta_data[f"UT.Strike_{idx}"] = ce_data["scrip"].iloc[0]
            meta_data[f"UT.Price_{idx}"] = ce_data.loc[pe_sl_time:, "close"].iloc[0]
            # meta_data[f"UT.Price_{idx}"] = ce_data.loc[pe_sl_time, "close"]
            meta_data[f"UT.SL.Time_{idx}"] = (
                sl_time.time() if sl_time != ce_data.index[-1] else None
            )
            meta_data[f"BPL_{idx}"] = 0
            meta_data[f"TT.PL.AT.SL_{idx}"] = round(pe_pnl, 2)
            meta_data[f"UT.PL.AT.SL_{idx}"] = (
                ce_price - ce_data.loc[pe_sl_time:, "close"].iloc[0] - (ce_price * 0.01)
            )
            meta_data[f"Total.PNL_{idx}"] = (
                meta_data[f"TT.PL.AT.SL_{idx}"] + meta_data[f"UT.PL.AT.SL_{idx}"] + ut_pnl
            )
            # print("ce sl hit at",sl_time)
            # print("pe pnl",pe_pnl,"ce pnl",ce_pnl,"ut pnl",ut_pnl)
    elif ce_sl_time is None and pe_sl_time is None:
        ce_pnl = ce_price - ce_data.iloc[-1]["close"] - (ce_price * 0.01)
        meta_data[f"CE.PNL_{idx}"] = round(ce_pnl, 2)
        meta_data[f"BPL_{idx}"] = ce_pnl + (
            pe_price - pe_data.iloc[-1]["close"] - (pe_price * 0.01)
        )
        meta_data[f"PE.PNL_{idx}"] = (
            pe_price - pe_data.iloc[-1]["close"] - (pe_price * 0.01)
        )
        meta_data[f"Total.PNL_{idx}"] = round(meta_data[f"BPL_{idx}"],2)
        # print("ce sl and pe sl not hit","pnl",ce_pnl,meta_data["PE.PNL"])
    elif pe_sl_time is None and ce_sl_time is not None:
        meta_data[f"CE.SL.Time_{idx}"] = ce_sl_time.time()
        data = pe_data[ce_sl_time:]
        close_price, pe_ut_price, sl_time = ut_check(data, ce_sl_time, ut_sl)
        ut_pnl = pe_ut_price - close_price
        pe_pnl = pe_price - pe_data.iloc[-1]['close'] - (pe_price * 0.01)
        ce_pnl = ce_price - ce_sl_price - (ce_price * 0.01)
        meta_data[f"CE.PNL_{idx}"] = round(ce_pnl, 2)
        meta_data[f"PE.PNL_{idx}"] = round(pe_pnl, 2)
        meta_data[f"UT.PNL_{idx}"] = round(ut_pnl, 2)
        meta_data[f"UT.Strike_{idx}"] = pe_scrip

        meta_data[f"UT.Price_{idx}"] = pe_data.loc[ce_sl_time:, "close"].iloc[0]
        meta_data[f"UT.SL.Time_{idx}"] = (
            sl_time.time() if sl_time != pe_data.index[-1] else None
        )
        meta_data[f"BPL_{idx}"] = 0
        meta_data[f"TT.PL.AT.SL_{idx}"] = round(ce_pnl, 2)
        meta_data[f"UT.PL.AT.SL_{idx}"] = (
            pe_price - pe_data.loc[ce_sl_time:, "close"].iloc[0] - (pe_price * 0.01)
        )
        meta_data[f"Total.PNL_{idx}"] = (
            meta_data[f"TT.PL.AT.SL_{idx}"] + meta_data[f"UT.PL.AT.SL_{idx}"] + ut_pnl
        )
    elif pe_sl_time is not None and ce_sl_time is None:
        meta_data[f"PE.SL.Time_{idx}"] = pe_sl_time.time()
        data = ce_data[pe_sl_time:]
        # print(ce_sl_time,pe_sl_time)
        close_price, ce_ut_price, sl_time = ut_check(data, pe_sl_time, ut_sl)
        # print("after ut check")
        ut_pnl = ce_ut_price - close_price
        ce_pnl = ce_price - ce_data.iloc[-1]["close"] - (ce_price * 0.01)
        pe_pnl = pe_price - pe_sl_price - (pe_price * 0.01)
        meta_data[f"CE.PNL_{idx}"] = round(ce_pnl, 2)
        meta_data[f"PE.PNL_{idx}"] = round(pe_pnl, 2)
        meta_data[f"UT.PNL_{idx}"] = round(ut_pnl, 2)
        meta_data[f"UT.Strike_{idx}"] = ce_scrip
        meta_data[f"UT.Price_{idx}"] = ce_data.loc[pe_sl_time, "close"]
        meta_data[f"UT.SL.Time_{idx}"] = (
            sl_time.time() if sl_time != ce_data.index[-1] else None
        )
        meta_data[f"BPL_{idx}"] = 0
        meta_data[f"TT.PL.AT.SL_{idx}"] = pe_pnl
        meta_data[f"UT.PL.AT.SL_{idx}"] = (
            ce_price - ce_data.loc[pe_sl_time, "close"] - (ce_price * 0.01)
        )
        meta_data[f"Total.PNL_{idx}"] = round((pe_pnl + meta_data[f"UT.PL.AT.SL_{idx}"] + ut_pnl), 2)

    return meta_data


def b120_intraday(
    option: pd.DataFrame, future: pd.DataFrame, start_time: pd.Timestamp, end_time: pd.Timestamp, sl: float|int, ut_sl: float|int, om: float, idx: int
):
    if om != 0:
        ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt = (
            get_strangle_strike(option, future, start_time, end_time, 50, om)
        )
    else:
        ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt = (
            get_straddle_strike(option, future, start_time, end_time)
        )

    if ce_scrip is None or pe_scrip is None or ce_price is None or pe_price is None:
        print(f"No valid strikes found for period {start_time} to {end_time}")
        return None

    # Calculate CE and PE stop loss prices as sl percentage above their entry prices
    ce_sl_price = ce_price + ((ce_price * sl) / 100)
    pe_sl_price = pe_price + ((pe_price * sl) / 100)
    option = option.loc[start_time:end_time]

    option = option = option[
        option.index > start_time
    ]  # to avoid entry time data in sl hit

    ce_data = option[option["scrip"] == ce_scrip]
    pe_data = option[option["scrip"] == pe_scrip]

    print("CE",ce_scrip,"PE", pe_scrip, current_dt)

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
        idx
    )

    return data

start_times_1 = pd.to_datetime(
    param_df.start_time_1.dropna().unique(), format="%H:%M:%S"
).time
end_times_1 = pd.to_datetime(param_df.end_time_1.dropna().unique(), format="%H:%M:%S").time
sls_1 = param_df.sl_1.dropna().unique()
ut_sls_1 = param_df.ut_sl_1.dropna().unique()
oms_1 = param_df.om_1.dropna().unique()

start_times_2 = pd.to_datetime(
    param_df.start_time_2.dropna().unique(), format="%H:%M:%S"
).time
end_times_2 = pd.to_datetime(param_df.end_time_2.dropna().unique(), format="%H:%M:%S").time
sls_2 = param_df.sl_2.dropna().unique()
ut_sls_2 = param_df.ut_sl_2.dropna().unique()
oms_2 = param_df.om_2.dropna().unique()


# ************** run for all files ****************
combination=[]

for future_path, option_path in zip(nifty_future_list, nifty_options_list):
    option = pd.read_pickle(option_path).set_index("date_time")
    fut = pd.read_pickle(future_path).set_index("date_time")
    option.index = pd.to_datetime(option.index)
    fut.index = pd.to_datetime(fut.index)
    option["date_time"] = option.index
    fut["date_time"] = fut.index

    file_date = fut.index[0].date()

    combinations = []

    for start_time_1, start_time_2 in zip(start_times_1, start_times_2):
        for end_time_1, end_time_2 in zip(end_times_1, end_times_2):
            for sl_1, sl_2 in zip(sls_1, sls_2):
                for ut_sl_1, ut_sl_2 in zip(ut_sls_1, ut_sls_2):
                    for om_1, om_2 in zip(oms_1, oms_2):

                        start_dt_1 = pd.to_datetime(
                                f"{file_date} {start_time_1.strftime('%H:%M:%S')}"
                            )
                        end_dt_1 = pd.to_datetime(
                                f"{file_date} {end_time_1.strftime('%H:%M:%S')}"
                            )

                        start_dt_2 = pd.to_datetime(
                                f"{file_date} {start_time_2.strftime('%H:%M:%S')}"
                            )
                        end_dt_2 = pd.to_datetime(
                                f"{file_date} {end_time_2.strftime('%H:%M:%S')}"
                            )

                        data_1 = b120_intraday(option,fut,start_dt_1,end_dt_1,sl_1,ut_sl_1,om_1,1)
                        data_2 = b120_intraday(option,fut,start_dt_2,end_dt_2,sl_2,ut_sl_2,om_2,2)
                        
                        if data_1 and data_2:
                            data_1.update(data_2)
                            data_1["Total.PNL"] = round((data_1["Total.PNL_1"] + data_1["Total.PNL_2"]), 2)
                            combinations.append(data_1)
                        elif data_1:
                            # Only data_1 is valid
                            data_1["Total.PNL"] = data_1["Total.PNL_1"]
                            combinations.append(data_1)
                        elif data_2:
                            # Only data_2 is valid
                            data_2["Total.PNL"] = data_2["Total.PNL_2"]
                            combinations.append(data_2)
                        else:
                            # Both are None, skip this combination or log it
                            print(f"Skipping combination: both periods returned None for date {file_date}")
                        
                        

    df = pd.DataFrame(combinations)
    os.makedirs(cwd / "B120G output", exist_ok=True)
    output_path = cwd / "B120G output"
    pd.DataFrame(combinations).to_csv(output_path / f"NIFTY {fut.index.date[0]} B120G.csv", index=False)

    combination.extend(combinations)
pd.DataFrame(combination).to_csv(cwd / "B120G output" / "NIFTY All Days B120G.csv", index=False)
