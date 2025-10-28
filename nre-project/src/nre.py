from pathlib import Path
import os
import pandas as pd
from typing import Literal
from tqdm import tqdm
from datetime import datetime

class NRE:
   
    def __init__(self, parameter_file: str):
        self.cwd = Path.cwd()
        self.nifty_future = Path("C:\\PICKLE\\Nifty Future")
        self.nifty_options = Path("C:\\PICKLE\\Nifty Options")
        self.param_df = pd.read_csv(parameter_file)

        self.start_date = pd.to_datetime(self.param_df.loc[0, "start_date"], dayfirst=True)
        self.end_date = pd.to_datetime(self.param_df.loc[0, "end_date"], dayfirst=True)

        self.nifty_options_list = []
        self.nifty_future_list = []
        self.processed_files = self.cwd / "processed_files"
        self.processed_files.mkdir(exist_ok=True)

    def get_file_date(self, file: Path) -> datetime:
        try:
            date_str = file.stem.split("_")[0]
            return datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return None

    def load_data(self):
        options_dict = {}
        for file in self.nifty_options.iterdir():
            if file.is_file():
                file_date = self.get_file_date(file)
                if file_date and self.start_date <= file_date <= self.end_date:
                    options_dict[file_date.date()] = file

        futures_dict = {}
        for file in self.nifty_future.iterdir():
            if file.is_file():
                file_date = self.get_file_date(file)
                if file_date and self.start_date <= file_date <= self.end_date:
                    futures_dict[file_date.date()] = file

        common_dates = sorted(set(options_dict.keys()) & set(futures_dict.keys()))
        self.nifty_options_list = [options_dict[d] for d in common_dates]
        self.nifty_future_list = [futures_dict[d] for d in common_dates]

    def get_one_om(self, fut, future_price=None, STEP=1000):
        future_price = fut['close'].iloc[0] if future_price is None else future_price
        return (int(future_price / STEP) * STEP) / 100

    def get_strangle_strike(self, fut, start_time, end_time, opt, gap=50, om=None, target=None, check_inverted=False, tf=1):
        valid_times = fut.loc[start_time:end_time].index
        for current_dt in valid_times:
            try:
                future_price = fut.loc[current_dt, 'close']
                one_om = self.get_one_om(fut, future_price)
                target = one_om * om if target is None else target
                target_od = opt[(opt.index == current_dt) & (opt['close'] >= target * tf)].sort_values(by=['close']).copy()

                ce_scrip = target_od.loc[target_od['scrip'].str.endswith('CE'), 'scrip'].iloc[0]
                pe_scrip = target_od.loc[target_od['scrip'].str.endswith('PE'), 'scrip'].iloc[0]

                ce_scrip_list = [ce_scrip, f"{int(ce_scrip[:-2]) - gap}CE", f"{int(ce_scrip[:-2]) + gap}CE"]
                pe_scrip_list = [pe_scrip, f"{int(pe_scrip[:-2]) - gap}PE", f"{int(pe_scrip[:-2]) + gap}PE"]

                call_list_prices, put_list_prices = [], []
                for z in range(3):
                    try:
                        call_list_prices.append(opt[(opt['date_time'] == current_dt) & (opt['scrip'] == ce_scrip_list[z])]['close'].iloc[0])
                    except:
                        call_list_prices.append(0)
                    try:
                        put_list_prices.append(opt[(opt['date_time'] == current_dt) & (opt['scrip'] == pe_scrip_list[z])]['close'].iloc[0])
                    except:
                        put_list_prices.append(0)

                call, put, min_diff = call_list_prices[0], put_list_prices[0], float('inf')
                target_2, target_3 = target * 2 * tf, target * 3

                diff = abs(put - call)
                required_call, required_put = None, None
                if (put + call >= target_2) & (min_diff > diff) & (put + call <= target_3):
                    min_diff = diff
                    required_call, required_put = call, put

                for i in range(1, 3):
                    if (min_diff > abs(put_list_prices[i] - call)) & (put_list_prices[i] + call >= target_2) & (put_list_prices[i] + call <= target_3):
                        min_diff = abs(put_list_prices[i] - call)
                        required_call, required_put = call, put_list_prices[i]
                    if (min_diff > abs(call_list_prices[i] - put)) & (call_list_prices[i] + put >= target_2) & (call_list_prices[i] + put <= target_3):
                        min_diff = abs(call_list_prices[i] - put)
                        required_call, required_put = call_list_prices[i], put

                ce_scrip, pe_scrip = ce_scrip_list[call_list_prices.index(required_call)], pe_scrip_list[put_list_prices.index(required_put)]
                ce_price, pe_price = opt[(opt['date_time'] == current_dt) & (opt['scrip'] == ce_scrip)]['close'].iloc[0], opt[(opt['date_time'] == current_dt) & (opt['scrip'] == pe_scrip)]['close'].iloc[0]

                if int(ce_scrip[:-2]) < int(pe_scrip[:-2]) and check_inverted:
                    self.get_straddle_strike(fut, current_dt, end_time, opt)
                else:
                    return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt

            except Exception as e:
                print("error occurred in get strangle strike", e)
                continue

        return None, None, None, None, None, None

    def get_straddle_strike(self, start_dt, end_dt, opt, fut, gap):
        valid_times = fut.loc[start_dt:end_dt].index
        for current_dt in valid_times:
            try:
                future_price = fut.loc[current_dt, 'close']
                round_future_price = round(future_price / gap) * gap
                ce_scrip, pe_scrip = f"{round_future_price}CE", f"{round_future_price}PE"
                ce_price, pe_price = opt[(opt['date_time'] == current_dt) & (opt['scrip'] == ce_scrip)]['close'].iloc[0], opt[(opt['date_time'] == current_dt) & (opt['scrip'] == pe_scrip)]['close'].iloc[0]
                syn_future = ce_price - pe_price + round_future_price
                round_syn_future = round(syn_future / gap) * gap

                ce_scrip_list = [f"{round_syn_future}CE", f"{round_syn_future + gap}CE", f"{round_syn_future - gap}CE"]
                pe_scrip_list = [f"{round_syn_future}PE", f"{round_syn_future + gap}PE", f"{round_syn_future - gap}PE"]

                scrip_index, min_value = None, float("inf")
                for i in range(3):
                    try:
                        ce_price = opt[(opt['date_time'] == current_dt) & (opt['scrip'] == ce_scrip_list[i])]['close'].iloc[0]
                        pe_price = opt[(opt['date_time'] == current_dt) & (opt['scrip'] == pe_scrip_list[i])]['close'].iloc[0]
                        diff = abs(ce_price - pe_price)
                        if min_value > diff:
                            min_value = diff
                            scrip_index = i
                    except:
                        pass
                ce_scrip, pe_scrip = ce_scrip_list[scrip_index], pe_scrip_list[scrip_index]
                ce_price, pe_price = opt[(opt['date_time'] == current_dt) & (opt['scrip'] == ce_scrip)]['close'].iloc[0], opt[(opt['date_time'] == current_dt) & (opt['scrip'] == pe_scrip)]['close'].iloc[0]

                return ce_scrip, pe_scrip, ce_price, pe_price, future_price, current_dt

            except (IndexError, KeyError, ValueError, TypeError):
                continue
            except Exception as e:
                print("get strike selection", e)
                continue
        return None, None, None, None, None, None

    def run_leg(self, data: pd.DataFrame, sl: int, sl_price, method: Literal["HL", "CC"], order_side: Literal['CE', 'PE']):
        total_pnl = 0
        price = data.iloc[0].close
        data = data.iloc[1:]
        sl_count = 0
        high, low = ('high', 'low') if method == 'HL' else ('close', 'close')
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

        while True:
            try:
                sl_time = data[data[high] >= sl_price].index[0]
            except:
                sl_time = None
            if sl_time is not None:
                if method == "HL":
                    pnl = price - sl_price - (0.01 * price)
                    total_pnl += pnl
                else:
                    pnl = price - data.loc[sl_time, 'close'] - (0.01 * price)
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
                sell_again_time = data[(data[low] <= price) & (data.index > sl_time)].index[0]
            except:
                sell_again_time = None

            if sell_again_time is not None:
                data = data[data.index > sell_again_time]

                leg_meta[f"{order_side}{sl_count}.Decay.Flag"] = True
                leg_meta[f"{order_side}{sl_count}.Decay.Time"] = str(sell_again_time)
            else:
                break

        return total_pnl, leg_meta

    def ner_both(self, option, future, strategy="NRE", symbol="NIFTY",
                 start_time="09:20:00", end_time="15:25:00",
                 method='CC', sl=20.0, resl=20.0, om=0,
                 trade_date="2022-01-04"):
        opt = option.loc[start_time:end_time]

        if om == 0:
            ce_scrip, pe_scrip, ce_price, pe_price, future_scrip, entry_time = self.get_straddle_strike(start_time, end_time, opt.copy(), future.copy(), 50)
        else:
            ce_scrip, pe_scrip, ce_price, pe_price, future_scrip, entry_time = self.get_strangle_strike(future.copy(), start_time, end_time, opt.copy(), om=om)

        ce_sl_price = ce_price * (1 + sl / 100)
        pe_sl_price = pe_price * (1 + sl / 100)

        ce_data = opt[(opt['scrip'] == ce_scrip)]
        pe_data = opt[(opt['scrip'] == pe_scrip)]
        ce_price = ce_data.loc[entry_time, 'close']
        pe_price = pe_data.loc[entry_time, 'close']

        ce_pnl, ce_meta = self.run_leg(ce_data, sl, ce_sl_price, method=method, order_side="CE")
        pe_pnl, pe_meta = self.run_leg(pe_data, sl, pe_sl_price, method=method, order_side="PE")

        metadata = {
            "P_Strategy": strategy,
            "P_Index": symbol,
            "P_StartTime": start_time,
            "P_EndTime": end_time,
            "P_OrderSide": "SELL",
            "P_Method": method,
            "P_SL": sl,
            "P_ReSL": resl,
            "P_OM": om,
            "Date": trade_date,
            "Day": pd.to_datetime(trade_date).day_name(),
            "DTE": ce_data.iloc[0].dte + 1,
            "EntryTime": start_time,
            "Future": future_scrip,
            "CE.Strike": ce_scrip,
            "CE.Price": ce_data.iloc[0].close,
            "CE.SL.Price": ce_sl_price if ce_data.iloc[0].close != ce_sl_price else None
        }
        pe = {
            "PE.Strike": pe_scrip,
            "PE.Price": pe_data.iloc[0].close,
            "PE.SL.Price": pe_sl_price if pe_data.iloc[0].close != pe_sl_price else None,
        }

        metadata.update(ce_meta)
        metadata.update(pe)
        metadata.update(pe_meta)

        return ce_pnl + pe_pnl, metadata

    def ner_both_weekly(self, option: pd.DataFrame, future: pd.DataFrame, strategy="NREW", symbol="NIFTY",
                        start_time="09:20:00", end_time="15:25:00",
                        method='CC', sl=20.0, resl=20.0, om=0):
        opt = option.loc[start_time:end_time]

        if om == 0:
            ce_scrip, pe_scrip, ce_price, pe_price, future_scrip, entry_time = self.get_straddle_strike(start_time, end_time, opt.copy(), future.copy(), 50)
        else:
            ce_scrip, pe_scrip, ce_price, pe_price, future_scrip, entry_time = self.get_strangle_strike(future.copy(), start_time, end_time, opt.copy(), om=om)

        ce_sl_price = ce_price * (1 + sl / 100)
        pe_sl_price = pe_price * (1 + sl / 100)

        ce_data = opt[(opt['scrip'] == ce_scrip)]
        pe_data = opt[(opt['scrip'] == pe_scrip)]
        ce_price = ce_data.loc[entry_time, 'close']
        pe_price = pe_data.loc[entry_time, 'close']

        ce_pnl, ce_meta = self.run_leg(ce_data, sl, ce_sl_price, method=method, order_side="CE")
        pe_pnl, pe_meta = self.run_leg(pe_data, sl, pe_sl_price, method=method, order_side="PE")

        metadata = {
            "P_Strategy": strategy,
            "P_Index": symbol,
            "P_StartTime": start_time.time(),
            "P_EndTime": end_time.time(),
            "P_OrderSide": "SELL",
            "P_Method": method,
            "P_SL": sl,
            "P_ReSL": resl,
            "P_OM": om,
            "Start.Date": start_time.date(),
            "End.Date": end_time.date(),
            "Start.DTE": ce_data.iloc[0].dte + 1,
            "End.DTE": 1,
            "DayCount": (end_time.date() - start_time.date()).days + 1,
            "EntryTime": start_time,
            "Future": future_scrip,
            "CE.Strike": ce_scrip,
            "CE.Price": ce_data.iloc[0].close,
            "CE.SL.Price": ce_sl_price if ce_data.iloc[0].close != ce_sl_price else None
        }
        pe = {
            "PE.Strike": pe_scrip,
            "PE.Price": pe_data.iloc[0].close,
            "PE.SL.Price": pe_sl_price if pe_data.iloc[0].close != pe_sl_price else None,
        }

        metadata.update(ce_meta)
        metadata.update(pe)
        metadata.update(pe_meta)

        return ce_pnl + pe_pnl, metadata

    def group_files_by_week(self):
        """Group files by expiry week based on DTE decreasing sequence."""
        weekly_groups = []
        current_group = []
        last_dte = None

        for fut_path, opt_path in zip(self.nifty_future_list, self.nifty_options_list):
            opt = pd.read_pickle(opt_path)
            dte = opt.iloc[0]['dte']
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

    def run_weekly(self):
        """Runs the weekly NRE strategy and saves results for each week."""
        self.load_data()
        start_times = pd.to_datetime(self.param_df.start_time.dropna().unique(), format="%H:%M:%S").time
        end_times = pd.to_datetime(self.param_df.end_time.dropna().unique(), format="%H:%M:%S").time
        methods = self.param_df.method.dropna().unique().tolist()
        sls = self.param_df.sl.dropna().unique().tolist()
        oms = self.param_df.om.dropna().unique().tolist()

        weekly_groups = self.group_files_by_week()
        weekly_folder = self.processed_files / "weekly_file"
        weekly_folder.mkdir(exist_ok=True)

        for group in weekly_groups:
            futs = []
            opts = []
            for fut_path, opt_path in group:
                fut = pd.read_pickle(fut_path)
                opt = pd.read_pickle(opt_path)
                futs.append(fut)
                opts.append(opt)
            fut_week = pd.concat(futs).set_index("date_time")
            opt_week = pd.concat(opts).set_index("date_time")
            fut_week.index = pd.to_datetime(fut_week.index)
            opt_week.index = pd.to_datetime(opt_week.index)
            opt_week['date_time'] = opt_week.index
            fut_week['date_time'] = fut_week.index

            entry_file = group[0][0]
            entry_date = pd.read_pickle(entry_file).iloc[0]['date_time']
            entry_date = pd.to_datetime(entry_date)
            exit_file = group[-1][0]
            exit_date = pd.read_pickle(exit_file).iloc[-1]['date_time']
            exit_date = pd.to_datetime(exit_date)
            combinations = []
            print(f"Processing week: {entry_date.date()} to {exit_date.date()}")
            for start_time in tqdm(start_times):
                for end_time in end_times:
                    for method in methods:
                        for sl in sls:
                            for om in oms:
                                start_dt = pd.to_datetime(f"{entry_date.date()} {start_time.strftime('%H:%M:%S')}")
                                end_dt = pd.to_datetime(f"{exit_date.date()} {end_time.strftime('%H:%M:%S')}")
                                pnl, rows = self.ner_both_weekly(
                                    opt_week.copy(), fut_week.copy(),
                                    start_time=start_dt, end_time=end_dt,
                                    method=method, sl=sl, resl=sl, om=om,
                                )
                                combinations.append(rows)

            combination = pd.DataFrame(combinations)
            combination.to_csv(f"{os.path.join(weekly_folder, 'NREW NIFTY ' + str(entry_date.date()))}.csv", index=False)
            print("file completed", entry_date.date())

    def run(self):
        self.load_data()
        start_times = pd.to_datetime(self.param_df.start_time.dropna().unique(), format="%H:%M:%S").time
        end_times = pd.to_datetime(self.param_df.end_time.dropna().unique(), format="%H:%M:%S").time

        methods = self.param_df.method.dropna().unique().tolist()
        sls = self.param_df.sl.dropna().unique().tolist()
        oms = self.param_df.om.dropna().unique().tolist()

        for future_path, option_path in zip(self.nifty_future_list, self.nifty_options_list):
            option = pd.read_pickle(option_path).set_index("date_time")
            fut = pd.read_pickle(future_path).set_index("date_time")
            option.index = pd.to_datetime(option.index)
            fut.index = pd.to_datetime(fut.index)
            option['date_time'] = option.index
            fut['date_time'] = fut.index

            file_date = fut.index[0].date()

            combinations = []
            for start_time in tqdm(start_times):
                for end_time in end_times:
                    for method in methods:
                        for sl in sls:
                            for om in oms:
                                start_dt = pd.to_datetime(f"{file_date} {start_time.strftime('%H:%M:%S')}")
                                end_dt = pd.to_datetime(f"{file_date} {end_time.strftime('%H:%M:%S')}")

                                pnl, rows = self.ner_both(option.copy(), fut.copy(), start_time=start_dt, end_time=end_dt,
                                                          method=method,
                                                          sl=sl, resl=sl,
                                                          om=om,
                                                          trade_date=option.index[0].date()
                                                          )
                                combinations.append(rows)

            combination = pd.DataFrame(combinations)
            combination.to_csv(f"{os.path.join(self.processed_files, str(option.index[0].date()))}.csv", index=False)

            print("file completed", future_path)

    def run_and_save_all(self, output_file="all_intraday_results.csv"):
        """Runs the strategy for all dates and saves all results in a single CSV."""
        self.load_data()
        start_times = pd.to_datetime(self.param_df.start_time.dropna().unique(), format="%H:%M:%S").time
        end_times = pd.to_datetime(self.param_df.end_time.dropna().unique(), format="%H:%M:%S").time

        methods = self.param_df.method.dropna().unique().tolist()
        sls = self.param_df.sl.dropna().unique().tolist()
        oms = self.param_df.om.dropna().unique().tolist()

        all_combinations = []
        for future_path, option_path in zip(self.nifty_future_list, self.nifty_options_list):
            option = pd.read_pickle(option_path).set_index("date_time")
            fut = pd.read_pickle(future_path).set_index("date_time")
            option.index = pd.to_datetime(option.index)
            fut.index = pd.to_datetime(fut.index)
            option['date_time'] = option.index
            fut['date_time'] = fut.index

            file_date = fut.index[0].date()

            for start_time in tqdm(start_times):
                for end_time in end_times:
                    for method in methods:
                        for sl in sls:
                            for om in oms:
                                start_dt = pd.to_datetime(f"{file_date} {start_time.strftime('%H:%M:%S')}")
                                end_dt = pd.to_datetime(f"{file_date} {end_time.strftime('%H:%M:%S')}")

                                pnl, rows = self.ner_both(option.copy(), fut.copy(), start_time=start_dt, end_time=end_dt,
                                                          method=method,
                                                          sl=sl, resl=sl,
                                                          om=om,
                                                          trade_date=option.index[0].date()
                                                          )
                                all_combinations.append(rows)

        all_df = pd.DataFrame(all_combinations)
        all_df.to_csv(output_file, index=False)
        print(f"All results saved to {output_file}")

    def run_and_save_all_weekly(self, output_file="all_weekly_results.csv"):
        """Runs the weekly NRE strategy for all weeks and saves all results in a single CSV."""
        self.load_data()
        start_times = pd.to_datetime(self.param_df.start_time.dropna().unique(), format="%H:%M:%S").time
        end_times = pd.to_datetime(self.param_df.end_time.dropna().unique(), format="%H:%M:%S").time
        methods = self.param_df.method.dropna().unique().tolist()
        sls = self.param_df.sl.dropna().unique().tolist()
        oms = self.param_df.om.dropna().unique().tolist()

        weekly_groups = self.group_files_by_week()
        all_combinations = []
        for group in weekly_groups:
            futs = []
            opts = []
            for fut_path, opt_path in group:
                fut = pd.read_pickle(fut_path)
                opt = pd.read_pickle(opt_path)
                futs.append(fut)
                opts.append(opt)
            fut_week = pd.concat(futs).set_index("date_time")
            opt_week = pd.concat(opts).set_index("date_time")
            fut_week.index = pd.to_datetime(fut_week.index)
            opt_week.index = pd.to_datetime(opt_week.index)
            opt_week['date_time'] = opt_week.index
            fut_week['date_time'] = fut_week.index

            entry_file = group[0][0]
            entry_date = pd.read_pickle(entry_file).iloc[0]['date_time']
            entry_date = pd.to_datetime(entry_date)
            exit_file = group[-1][0]
            exit_date = pd.read_pickle(exit_file).iloc[-1]['date_time']
            exit_date = pd.to_datetime(exit_date)
            for start_time in tqdm(start_times):
                for end_time in end_times:
                    for method in methods:
                        for sl in sls:
                            for om in oms:
                                start_dt = pd.to_datetime(f"{entry_date.date()} {start_time.strftime('%H:%M:%S')}")
                                end_dt = pd.to_datetime(f"{exit_date.date()} {end_time.strftime('%H:%M:%S')}")
                                pnl, rows = self.ner_both_weekly(
                                    opt_week.copy(), fut_week.copy(),
                                    start_time=start_dt, end_time=end_dt,
                                    method=method, sl=sl, resl=sl, om=om,
                                )
                                all_combinations.append(rows)

        all_df = pd.DataFrame(all_combinations)
        all_df.to_csv(output_file, index=False)
        print(f"All weekly results saved to {output_file}")
    
if __name__ == "__main__":
    nre = NRE("parameter.csv")
    # nre.run()  # Comment this out if you only want the combined file
    # nre.run_and_save_all("all_results.csv")
    # nre.run_weekly()
    nre.run_and_save_all_weekly("all_weekly_results.csv")


