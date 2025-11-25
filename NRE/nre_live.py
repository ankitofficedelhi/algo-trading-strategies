import pandas as pd
from datetime import datetime
from pathlib import Path
import requests
from tqdm import gui, tqdm
import os
from time import sleep
from typing import Literal


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
    df["date_time"] = pd.to_datetime(df["date_time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
    df["date_time"] = pd.to_datetime(df["date_time"])

    
    return df

def get_one_om(fut, future_price=None, STEP=1000):
    future_price = fut["close"].iloc[0] if future_price is None else future_price
    return (int(future_price / STEP) * STEP) / 100

def get_strangle_strike(
    fut,
    start_time,
    end_time,
    gap=50,
    om=None,
    target=None,
    check_inverted=False,
    tf=1,
    cookies: str = "",
):

    valid_times = fut.loc[start_time:end_time].index
    for current_dt in valid_times:
        try:
            
            future_price = fut.loc[current_dt, "close"]            
            one_om = get_one_om(fut, future_price)
            target = one_om * om if target is None else target
            
            round_future_price = round(future_price / gap) * gap
            scrip_base = round_future_price
            scrip_list = [
                            f"{scrip_base}",
                            f"{scrip_base - gap}",
                            f"{scrip_base - 2*gap}",
                            f"{scrip_base + gap}",
                            f"{scrip_base + 2*gap}"
                        ]
            instrument_file = pd.read_csv("https://api.kite.trade/instruments")
            instrument_file = instrument_file[instrument_file['name'] == 'NIFTY'].copy()
            instrument_file['expiry'] = pd.to_datetime(instrument_file['expiry'])
            instrument_file = instrument_file[instrument_file['expiry'] == instrument_file['expiry'].min()]

            ce_data = pd.DataFrame()
            pe_data = pd.DataFrame()
            
            for ce_scrip in scrip_list:
                try:
                    ce_token = int(instrument_file.loc[instrument_file['tradingsymbol'].str.endswith(f"{ce_scrip}CE"), 'instrument_token'].iloc[0])
                    ce_opt = fetch_candle_data(ce_token, current_dt.date(), current_dt.date(), cookies)
                    ce_opt['scrip'] = f"{ce_scrip}CE"
                    ce_data = pd.concat([ce_data, ce_opt])
                    sleep(2)
                except:
                    continue
            for pe_scrip in scrip_list:
                try:
                    pe_token = int(instrument_file.loc[instrument_file['tradingsymbol'].str.endswith(f"{pe_scrip}PE"), 'instrument_token'].iloc[0])
                    pe_opt = fetch_candle_data(pe_token, current_dt.date(), current_dt.date(), cookies)
                    pe_opt['scrip'] = f"{pe_scrip}PE"
                    pe_data = pd.concat([pe_data, pe_opt])
                    sleep(2)
                except:
                    continue
            opt = pd.concat([ce_data, pe_data])
            opt.index = pd.to_datetime(opt['date_time'])

            print("----------------------------------------------")

            target_od = opt[(opt.index == current_dt) & (opt["close"] >= target * tf)].copy()
            if target_od.empty:
  
                print(f"[{current_dt}] No strikes >= target ({target*tf}). Relaxing filter.")
                target_od = opt[opt.index == current_dt].copy()

            if target_od.empty:
                print(f"[{current_dt}] No option data for this timestamp. Skipping.")
                continue
            
            ce_rows = target_od[target_od["scrip"].str.endswith("CE")]
            pe_rows = target_od[target_od["scrip"].str.endswith("PE")]
            
            if ce_rows.empty or pe_rows.empty:
                print(f"[{current_dt}] Missing CE or PE after filtering. Available scrips:", target_od["scrip"].unique())
                continue  # skip this minute
            
            ce_scrip = ce_rows.sort_values("close").iloc[0]["scrip"]
            pe_scrip = pe_rows.sort_values("close").iloc[0]["scrip"]
            
            print("ce scrip,pe scrip",ce_scrip,pe_scrip)
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
            # print("CE Scrip List:", ce_scrip_list)
            # print("PE Scrip List:", pe_scrip_list)
            call_list_prices, put_list_prices = [], []
            opt['date_time'] = pd.to_datetime(opt['date_time'])
            opt.index = opt['date_time']
            for ce_candidate in ce_scrip_list:
                try:
                    row = opt[(opt['date_time'] == current_dt) & (opt['scrip'] == ce_candidate)]
                    if row.empty:
                        # Fetch missing CE
                        try:
                            ce_token = int(
                                instrument_file.loc[
                                    instrument_file['tradingsymbol'].str.endswith(ce_candidate),
                                    'instrument_token'
                                ].iloc[0]
                            )
                            ce_new = fetch_candle_data(ce_token, current_dt.date(), current_dt.date(), cookies)
                            ce_new['scrip'] = ce_candidate
                            opt = pd.concat([opt, ce_new], ignore_index=True)
                            opt['date_time'] = pd.to_datetime(opt['date_time'])
                            opt.index = opt['date_time']
                            row = opt[(opt['date_time'] == current_dt) & (opt['scrip'] == ce_candidate)]
                        except Exception as e:
                            call_list_prices.append(0)
                            continue
                    call_list_prices.append(row['close'].iloc[0])
                except:
                    call_list_prices.append(0)

            for pe_candidate in pe_scrip_list:
                try:
                    row = opt[(opt['date_time'] == current_dt) & (opt['scrip'] == pe_candidate)]
                    if row.empty:
                        # Fetch missing PE
                        try:
                            pe_token = int(
                                instrument_file.loc[
                                    instrument_file['tradingsymbol'].str.endswith(pe_candidate),
                                    'instrument_token'
                                ].iloc[0]
                            )
                            pe_new = fetch_candle_data(pe_token, current_dt.date(), current_dt.date(), cookies)
                            pe_new['scrip'] = pe_candidate
                            opt = pd.concat([opt, pe_new], ignore_index=True)
                            opt['date_time'] = pd.to_datetime(opt['date_time'])
                            opt.index = opt['date_time']
                            row = opt[(opt['date_time'] == current_dt) & (opt['scrip'] == pe_candidate)]
                        except Exception as e:
                            put_list_prices.append(0)
                            continue
                    put_list_prices.append(row['close'].iloc[0])
                except:
                    put_list_prices.append(0)


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

def run_leg(
    data: pd.DataFrame,
    sl: int,
    sl_price,
    method: Literal["HL", "CC"],
    order_side: Literal["CE", "PE"],
    re_entry=1,
):
    """Runs one option leg (CE or PE) and returns metadata updates."""
    total_pnl = 0
    price = data.iloc[0].close
    data = data.iloc[1:]
    sl_count = 0
    high, low = ("high", "low") if method == "HL" else ("close", "close")
    leg_meta = {}
    for i in range(8):
        if i != 0:
            leg_meta[f"{order_side}{i}.Decay.Flag"] = False
            leg_meta[f"{order_side}{i}.Decay.Time"] = ""
        leg_meta[f"{order_side}{i}.SL.Flag"] = False
        leg_meta[f"{order_side}{i}.SL.Time"] = ""
        leg_meta[f"{order_side}{i}.PNL"] = 0
    if sl == 0:
        pnl = price - data.iloc[-1].close - (0.01 * price)
        leg_meta[f"{order_side}{0}.PNL"] = pnl
        return pnl, leg_meta
    
    for _ in range(re_entry):
    # while True:
        try:
            sl_time = data[data[high] >= sl_price].index[0]
        except:
            sl_time = None
        if sl_time is not None:
            if method == "HL":
                pnl = price - sl_price - (0.01 * price)
                total_pnl += pnl
            else:
                pnl = price - data.loc[sl_time, "close"] - (0.01 * price)
                total_pnl += pnl

            leg_meta[f"{order_side}{sl_count}.SL.Flag"] = True
            leg_meta[f"{order_side}{sl_count}.SL.Time"] = str(sl_time)
            leg_meta[f"{order_side}{sl_count}.PNL"] = pnl
            sl_count += 1

        if sl_time is None:
            if len(data.index) == 0:
                print("No data left for exit, skipping leg.")
                break
            exit_time = data.index[-1]
            pnl = price - data.loc[exit_time].close - (0.01 * price)
            total_pnl += pnl

            leg_meta[f"{order_side}{sl_count}.PNL"] = pnl
            break
        try:
            sell_again_time = data[(data[low] <= price) & (data.index > sl_time)].index[
                0
            ]
        except:
            sell_again_time = None

        if sell_again_time is not None:
            data = data[data.index > sell_again_time]

            leg_meta[f"{order_side}{sl_count}.Decay.Flag"] = True
            leg_meta[f"{order_side}{sl_count}.Decay.Time"] = str(sell_again_time)
        else:
            break

    return total_pnl, leg_meta, sl_count

def nre_live(sl, OM, start_time, end_time, method="HL",re_entry = 1,cookies=""):
    instrument_file = pd.read_csv("https://api.kite.trade/instruments")
    fut_df =instrument_file[(instrument_file["instrument_type"] == "FUT")& (instrument_file["name"]=="NIFTY")].copy()
    fut_df["expiry"] = pd.to_datetime(fut_df["expiry"])
    fut_df = fut_df[fut_df["expiry"] == fut_df["expiry"].min()]
    fut_token=int(fut_df["instrument_token"].values[0])
    fut = fetch_candle_data(fut_token, start_time.date(), end_time.date(), cookies)
    fut["date_time"] = pd.to_datetime(fut["date_time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
    fut = fut.set_index("date_time",  drop=True)
    fut.index = pd.to_datetime(fut.index)
    
    sd = 0.0
    om = 0.0
    if isinstance(OM, str):
        if OM.lower().endswith("sd"):
            sd = float(OM[:-2])
            om = 0.0
        elif OM.lower().endswith("s"):
            sd = float(OM[:-1])
            om = 0.0
        else:
            om = float(OM)
            sd = 0.0
    else:
        om = float(OM)
        sd = 0.0
    ce_scrip, pe_scrip, ce_price, pe_price, future_scrip, entry_time = (
            get_straddle_strike(fut.copy(),start_time, end_time,50,sd =sd,cookies=cookies)
        ) if om == 0.0 else get_strangle_strike(fut,start_time,end_time,50,om,cookies=cookies)
    
    ce_sl_price = ce_price * (1 + sl / 100)
    pe_sl_price = pe_price * (1 + sl / 100)

    instrument_file = pd.read_csv("https://api.kite.trade/instruments")
    instrument_file = instrument_file[instrument_file['name'] == 'NIFTY'].copy()
    instrument_file['expiry'] = pd.to_datetime(instrument_file['expiry'])
    instrument_file = instrument_file[instrument_file['expiry'] == instrument_file['expiry'].min()]
    ce_token = int(instrument_file.loc[instrument_file['tradingsymbol'].str.endswith(ce_scrip), 'instrument_token'].iloc[0])
    pe_token = int(instrument_file.loc[instrument_file['tradingsymbol'].str.endswith(pe_scrip), 'instrument_token'].iloc[0])
    
    while True:
        
        ce_data = fetch_candle_data(ce_token, entry_time.date(), entry_time.date(), cookies)
        pe_data = fetch_candle_data(pe_token, entry_time.date(), entry_time.date(), cookies)
        ce_data["date_time"] = pd.to_datetime(ce_data["date_time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
        pe_data["date_time"] = pd.to_datetime(pe_data["date_time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
        ce_data['date_time'] = pd.to_datetime(ce_data['date_time'])
        pe_data['date_time'] = pd.to_datetime(pe_data['date_time'])
        ce_data = ce_data.set_index("date_time",  drop=True)
        pe_data = pe_data.set_index("date_time",  drop=True)
        ce_data = ce_data[ce_data.index >= entry_time]
        pe_data = pe_data[pe_data.index >= entry_time]
        print("ce_price",ce_price,"pe_price",pe_price)
        print("ce_sl_price",ce_sl_price,"pe_sl_price",pe_sl_price)
        print("sl",sl)
        print("scripts:", ce_scrip, pe_scrip)
        print("open high low close")
        print("ce data",ce_data.iloc[-1][['open','high','low','close']].values,"pe data",pe_data.iloc[-1][['open','high','low','close']].values)
        ce_pnl, ce_meta, ce_sl_count = run_leg(ce_data, sl, ce_sl_price, method=method, order_side="CE",re_entry=re_entry)
        print("CE PNL:", round(ce_pnl,2))
        ce_sl_hit = False
        pe_sl_hit = False
        for count in range(ce_sl_count+1):
            if ce_meta[f"CE{count}.SL.Flag"]==True:
                print(f"CE SL hit at {ce_meta[f'CE{count}.SL.Time']}")
                ce_sl_hit = True
        pe_pnl, pe_meta, pe_sl_count = run_leg(pe_data, sl, pe_sl_price, method=method, order_side="PE",re_entry=re_entry)
        print("PE PNL:", round(pe_pnl,2),"pe_sl_hit times", pe_sl_count)
        total_pnl = ce_pnl + pe_pnl
        print("Total PNL:", round(total_pnl,2))
        for count in range(pe_sl_count+1):
            if pe_meta[f"PE{count}.SL.Flag"]==True:
                print(f"PE SL hit at {pe_meta[f'PE{count}.SL.Time']}")
                # pe_sl_hit = True
        print(start_time.time(), datetime.now().time())
        print("--------------------------------------------------")
        if end_time.time()<=datetime.now().time():
            break
        elif ce_sl_hit and pe_sl_hit:
            print("Not re-entering the trade as both SLs are hit.")
            break
        elif  end_time.time()>=datetime.now().time():
            print("Re-entering the trade...")
            sleep(30)
            continue
        
        
    return total_pnl, ce_meta, pe_meta

param_df=pd.read_csv("live_param.csv")
date = param_df["date"].dropna().values[0]
re_entry = param_df["re_entry"].dropna().values[0]
sl = param_df["sl"].dropna().values[0]
om = param_df["om"].dropna().values[0]
start_time = pd.to_datetime(param_df["start_time"].dropna().values[0], dayfirst=True)
end_time = pd.to_datetime(param_df["end_time"].dropna().values[0], dayfirst=True)
method = param_df["method"].dropna().values[0]
cookies = "enctoken vrmUiewGd4//6mDIhJkEhURevIck7c5v18OvagB/z7EWD1t96Ia/Zd8GuODZ1vOAEt8xAQJiL7ARdETZdW/qukxYFSVL8PONq4+A5CBqpuXzrnHJyKWicA=="
start_time = pd.to_datetime(f"{date} {start_time.time()}",dayfirst=True)
end_time = pd.to_datetime(f"{date} {end_time.time()}",dayfirst=True)


_,ce_meta, pe_meta = nre_live(
    sl=sl,
    OM=om,
    start_time=start_time,
    end_time=end_time,
    method=method,
    re_entry=re_entry+1,
    cookies=cookies
    )
merged = {**ce_meta, **pe_meta}
df = pd.DataFrame([merged]).to_csv(f"nre_live_result_{date}.csv", index=False)