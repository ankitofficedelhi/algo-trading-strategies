import pandas as pd
from datetime import datetime
from pathlib import Path
import requests
from tqdm import gui, tqdm
import os
from time import sleep


cwd = Path.cwd()
param_df = pd.read_csv("parameter.csv")
start_date = pd.to_datetime(param_df.loc[0, "start_date"], dayfirst=True)
end_date = pd.to_datetime(param_df.loc[0, "end_date"], dayfirst=True)

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

def fetch_candle_data(token, from_date, to_date, cookies):
    headers = {
        "authorization": cookies
    }

    url = f"https://kite.zerodha.com/oms/instruments/historical/{token}/minute?user_id=ZGA974&oi=1&from={from_date}&to={to_date}"

    response = requests.get(url, headers=headers)

    response = response.json()
    # print(response)
    df = pd.DataFrame(response['data']['candles'])
    df.columns = ['date_time', 'open', 'high', 'low', 'close', 'volume', 'openinterest']

    return df

def get_straddle_strike(fut: pd.DataFrame,start_dt: pd.Timedelta,end_dt: pd.Timedelta,gap = 50,sd: int = 0, cookies: str="", SDroundoff: bool = False):

    valid_times = fut.loc[start_dt:end_dt].index
    for current_dt in valid_times:
        try:
            # print(f"Processing for time: {current_dt}")
            future_price = fut.loc[current_dt, "close"]
            round_future_price = round(future_price / gap) * gap
            ce_scrip, pe_scrip = f"{round_future_price}CE", f"{round_future_price}PE"
            # print("1",ce_scrip, pe_scrip)
            instrument_file = pd.read_csv("https://api.kite.trade/instruments")
            instrument_file = instrument_file[instrument_file['name'] == 'NIFTY'].copy()
            instrument_file['expiry'] = pd.to_datetime(instrument_file['expiry'])
            instrument_file = instrument_file[instrument_file['expiry'] == instrument_file['expiry'].min()]
            ce_token = int(instrument_file.loc[instrument_file['tradingsymbol'].str.endswith(ce_scrip), 'instrument_token'].iloc[0])
            pe_token = int(instrument_file.loc[instrument_file['tradingsymbol'].str.endswith(pe_scrip), 'instrument_token'].iloc[0])
            # print(ce_token, pe_token)
            
            ce_opt = fetch_candle_data(ce_token, current_dt.date(), current_dt.date(), cookies)
            pe_opt = fetch_candle_data(pe_token, current_dt.date(), current_dt.date(), cookies)
            
            # print(ce_opt)
            ce_opt["date_time"] = pd.to_datetime(ce_opt["date_time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            pe_opt["date_time"] = pd.to_datetime(pe_opt["date_time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            ce_opt['date_time'] = pd.to_datetime(ce_opt['date_time'])
            pe_opt['date_time'] = pd.to_datetime(pe_opt['date_time'])
            ce_price, pe_price = (
                ce_opt[
                    (ce_opt["date_time"] == current_dt)
                ].close.iloc[0],
                pe_opt[
                    (pe_opt["date_time"] == current_dt)
                ].close.iloc[0],
            )
            # print("1",ce_price, pe_price)

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
                    ce_token = int(instrument_file.loc[instrument_file['tradingsymbol'].str.endswith(ce_scrip_list[i]), 'instrument_token'].iloc[0])
                    pe_token = int(instrument_file.loc[instrument_file['tradingsymbol'].str.endswith(pe_scrip_list[i]), 'instrument_token'].iloc[0])
                    
                    
                    ce_opt = fetch_candle_data(ce_token, current_dt.date(), current_dt.date(), cookies)
                    pe_opt = fetch_candle_data(pe_token, current_dt.date(), current_dt.date(), cookies)
                    ce_opt["date_time"] = pd.to_datetime(ce_opt["date_time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    pe_opt["date_time"] = pd.to_datetime(pe_opt["date_time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    ce_opt['date_time'] = pd.to_datetime(ce_opt['date_time'])
                    pe_opt['date_time'] = pd.to_datetime(pe_opt['date_time'])
                    
                    ce_price, pe_price = (
                    ce_opt[(ce_opt["date_time"] == current_dt)].close.iloc[0],
                    pe_opt[(pe_opt["date_time"] == current_dt)].close.iloc[0],
                    )
                    diff = abs(ce_price - pe_price)
                    if min_value > diff:
                        min_value = diff
                        scrip_index = i
                except:
                    pass
            ce_scrip, pe_scrip = ce_scrip_list[scrip_index], pe_scrip_list[scrip_index]
            
            ce_token = int(instrument_file.loc[instrument_file['tradingsymbol'].str.endswith(ce_scrip), 'instrument_token'].iloc[0])
            pe_token = int(instrument_file.loc[instrument_file['tradingsymbol'].str.endswith(pe_scrip), 'instrument_token'].iloc[0])
                    
                    
            ce_opt = fetch_candle_data(ce_token, current_dt.date(), current_dt.date(), cookies)
            pe_opt = fetch_candle_data(pe_token, current_dt.date(), current_dt.date(), cookies)
            ce_opt["date_time"] = pd.to_datetime(ce_opt["date_time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            pe_opt["date_time"] = pd.to_datetime(pe_opt["date_time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            ce_opt['date_time'] = pd.to_datetime(ce_opt['date_time'])
            pe_opt['date_time'] = pd.to_datetime(pe_opt['date_time'])
            ce_price, pe_price = (
            ce_opt[(ce_opt["date_time"] == current_dt)].close.iloc[0],
            pe_opt[(pe_opt["date_time"] == current_dt)].close.iloc[0],
            )
            
            # sd_range = 0
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
                ce_token = int(instrument_file.loc[instrument_file['tradingsymbol'].str.endswith(ce_scrip), 'instrument_token'].iloc[0])
                pe_token = int(instrument_file.loc[instrument_file['tradingsymbol'].str.endswith(pe_scrip), 'instrument_token'].iloc[0])
                        
                    
                ce_opt = fetch_candle_data(ce_token, current_dt.date(), current_dt.date(), cookies)
                pe_opt = fetch_candle_data(pe_token, current_dt.date(), current_dt.date(), cookies)
                
                ce_price, pe_price = (
                ce_opt[(ce_opt["date_time"] == current_dt)].close.iloc[0],
                pe_opt[(pe_opt["date_time"] == current_dt)].close.iloc[0],
                )

                return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt
            return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt
        except (TypeError, ValueError, KeyError, IndexError):
            print("error occured in get straddle strike")
            continue

    return None, None, None, None, None, None

def ut_check(data, sl_time, ut_sl):
    price = data.loc[sl_time, "close"]
    ut_sl_price = price + ((price * ut_sl) / 100)
    data = data[data.index > sl_time]

    # Check if data is empty after filtering
    if data.empty:
        return price, price, sl_time

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
        # "DTE": ce_data["dte"].iloc[0] + 1,
        "EntryTime": current_dt.time(),
        "Future": future_price,
        "CE.Strike": ce_scrip,
        "CE.Open": ce_price,
        "CE.High": ce_data["high"].max(),
        "CE.Low": ce_data["low"].min(),
        "CE.Close": ce_data["close"].iloc[-1],
        "CE.SL.Flag": False,  # update in sl time compairison
        "CE.SL.Time": None,
        "CE.PNL": None,
        "PE.Strike": pe_scrip,
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
            meta_data["UT.Strike"] = pe_scrip
            meta_data["UT.Open"] = pe_data.loc[ce_sl_time, "close"]
            meta_data["UT.High"] = pe_data.loc[ce_sl_time:exit_time, "high"].max()
            meta_data["UT.Low"] = pe_data.loc[ce_sl_time:exit_time, "low"].min()
            meta_data["UT.Close"] = pe_data.loc[exit_time, "close"] if exit_time in pe_data.index else pe_data["close"].iloc[-1]
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
            print("pe price",pe_price,"pe at st time",pe_data.loc[ce_sl_time,'close'])
            print("ce sl hit at",sl_time)
        else:    
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
            print("pe sl hit at",pe_sl_time)
            print("ut sl hit at",sl_time)
            print("pe pnl",pe_pnl,"ce pnl",ce_pnl,"ut pnl",ut_pnl)
    elif ce_sl_time is None and pe_sl_time is None:
        exit_time = ce_data.index[-1]
        ce_pnl = ce_price - ce_data.loc[exit_time, "close"] - (ce_price * 0.01)
        meta_data["CE.PNL"] = ce_pnl
        pe_exit_time = pe_data.index[-1]
        meta_data["BPL"] = ce_pnl + (
            pe_price - pe_data.loc[pe_exit_time, "close"] - (pe_price * 0.01)
        )
        meta_data["PE.PNL"] = (
            pe_price - pe_data.loc[pe_exit_time, "close"] - (pe_price * 0.01)
        )
        meta_data["Total.PNL"] = meta_data["BPL"]
        print("ce sl and pe sl not hit","pnl",ce_pnl,meta_data["PE.PNL"])
    elif pe_sl_time is None and ce_sl_time is not None:
        exit_time = pe_data.index[-1]
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
        print("utsl time",meta_data["UT.SL.Time"])
        meta_data["BPL"] = 0
        meta_data["TT.PL.AT.SL"] = ce_pnl
        meta_data["UT.PL.AT.SL"] = (
            pe_price - pe_data.loc[ce_sl_time, "close"] - (pe_price * 0.01)
        )
        meta_data["Total.PNL"] = (
            meta_data["TT.PL.AT.SL"] + meta_data["UT.PL.AT.SL"] + ut_pnl
        )
    elif pe_sl_time is not None and ce_sl_time is None:
        exit_time = pe_data.index[-1]
        meta_data["PE.SL.Flag"] = True
        meta_data["PE.SL.Time"] = pe_sl_time.time()
        data = ce_data[pe_sl_time:]
        print(ce_sl_time,pe_sl_time)
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
     future, start_time, end_time, method, sl, ut_sl, om, trade_date
):
    cookies = "enctoken o3FsOISPx22y1s6s6VKpY1UIW7CLjnrtqKvBYsGuJsvEd3Bvj0OD+j6D6n+xeu+B7ZyhkhvVhu2nYL+aGZ8PMLAFN/wQFPQqf5RVWmdaGPKUrTdaL2cADw=="
    ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt = (
            get_straddle_strike( future, start_time, end_time,gap=50,cookies=cookies)
        )
    
    # Calculate CE and PE stop loss prices as sl percentage above their entry prices
    ce_sl_price = ce_price + ((ce_price * sl) / 100)
    pe_sl_price = pe_price + ((pe_price * sl) / 100)
    # option = option.loc[start_time:end_time]

    # option = option = option[
    #     option.index > start_time
    # ]  # to avoid entry time data in sl hit

    instrument_file = pd.read_csv("https://api.kite.trade/instruments")
    instrument_file = instrument_file[instrument_file['name'] == 'NIFTY'].copy()
    instrument_file['expiry'] = pd.to_datetime(instrument_file['expiry'])
    instrument_file = instrument_file[instrument_file['expiry'] == instrument_file['expiry'].min()]
    ce_token = int(instrument_file.loc[instrument_file['tradingsymbol'].str.endswith(ce_scrip), 'instrument_token'].iloc[0])
    pe_token = int(instrument_file.loc[instrument_file['tradingsymbol'].str.endswith(pe_scrip), 'instrument_token'].iloc[0])
    # print(ce_token, pe_token)
            
    while True:
                
        ce_data = fetch_candle_data(ce_token, current_dt.date(), current_dt.date(), cookies)
        pe_data = fetch_candle_data(pe_token, current_dt.date(), current_dt.date(), cookies)
                
        ce_data["date_time"] = pd.to_datetime(ce_data["date_time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
        ce_data = ce_data.set_index("date_time",  drop=True)
        ce_data.index = pd.to_datetime(ce_data.index)
        pe_data["date_time"] = pd.to_datetime(pe_data["date_time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
        pe_data = pe_data.set_index("date_time",  drop=True)
        pe_data.index = pd.to_datetime(pe_data.index)
        ce_data = ce_data[ce_data.index >= current_dt]
        pe_data = pe_data[pe_data.index >= current_dt]

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
        
        print("ce price",ce_data["high"].iloc[-1],"pe price",pe_data["high"].iloc[-1])
        print("pnl",data["CE.PNL"] + data["PE.PNL"])
        print("")
        
        if data["CE.SL.Flag"] or data["PE.SL.Flag"]:
            print("sl hit at",data["CE.SL.Time"] if data["CE.SL.Flag"] else data["PE.SL.Time"])
            if data["UT.SL.Flag"]:
                print("ut sl hit at",data["UT.SL.Time"])
                print("total pnl",data["Total.PNL"])
            break
        print("ce price",ce_data["high"].iloc[-1],"pe price",pe_data["high"].iloc[-1])
        print("pnl",data["CE.PNL"] + data["PE.PNL"])
        
        # current_dt += pd.Timedelta(minutes=1)
        if end_time.time() <= datetime.now().time():
            break
        sleep(60)
    return data

def b120_backtest():
    instrument_file = pd.read_csv("https://api.kite.trade/instruments")
    fut_df =instrument_file[(instrument_file["instrument_type"] == "FUT")& (instrument_file["name"]=="NIFTY")].copy()
    fut_df["expiry"] = pd.to_datetime(fut_df["expiry"])
    fut_df = fut_df[fut_df["expiry"] == fut_df["expiry"].min()]
    fut_token=int(fut_df["instrument_token"].values[0])
    cookies = "enctoken o3FsOISPx22y1s6s6VKpY1UIW7CLjnrtqKvBYsGuJsvEd3Bvj0OD+j6D6n+xeu+B7ZyhkhvVhu2nYL+aGZ8PMLAFN/wQFPQqf5RVWmdaGPKUrTdaL2cADw=="
    from_date = "2025-11-14"
    to_date = "2025-11-14"
    fut = fetch_candle_data(fut_token, from_date, to_date, cookies)
    fut["date_time"] = pd.to_datetime(fut["date_time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
    fut = fut.set_index("date_time",  drop=True)
    fut.index = pd.to_datetime(fut.index)
    
    entry_time = pd.to_datetime(f"{from_date} 12:20:00")
    exit_time = pd.to_datetime(f"{from_date} 15:29:00")
    # sl = param_df
    data = b120_intraday(fut,entry_time,exit_time,"HL",sl=20,ut_sl=30,om=0,trade_date=from_date)

        
    os.makedirs(cwd / "B120 output", exist_ok=True)
    output_path = cwd / "B120 output" / f"B120_Backtest_{from_date}_to_{to_date}.csv"
    data_df = pd.DataFrame([data])
    data_df.to_csv(output_path, index=False)
    
    
    
if __name__ == "__main__":
    b120_backtest()