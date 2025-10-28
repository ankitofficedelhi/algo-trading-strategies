import pandas as pd
def get_one_om(future_price,fpt,step):
    future_price = fpt['close'].iloc[0] if future_price is None else future_price

    return ((int(future_price/step)*step)/100)

def get_strangle_strike(start_dt,end_dt,opt:pd.DataFrame,fpt:pd.DataFrame,step:int,om=0,target = False,tf = 1):

    valid_times = fpt.loc[start_dt:end_dt].index
    for current_dt in valid_times:
        # om , target ,tf= 1 , 1,1
        future_price = fpt.loc[current_dt,'close']
        one_om = get_one_om(future_price,fpt,step)
        target = one_om * om if target is None else target
        # opt[(opt.index==current_dt)]
        target_od = opt[(opt.index==current_dt)&(opt['close']>=target*tf)].sort_values(by=['close']).copy()

        ce_scrip = target_od.loc[target_od['scrip'].str.endswith('CE'),'scrip'].iloc[0]
        pe_scrip = target_od.loc[target_od['scrip'].str.endswith('PE'),'scrip'].iloc[0]
        ce_scrip_list = [ce_scrip,f"{int(ce_scrip[:-2])-50}CE",f"{int(ce_scrip[:-2])+50}CE"]
        pe_scrip_list = [pe_scrip,f"{int(pe_scrip[:-2])-50}PE",f"{int(pe_scrip[:-2])+50}PE"]

        call_list_prices, put_list_prices = [], []
        for z in range(3):
            try:
                call_list_prices.append(opt[opt['scrip'] == ce_scrip_list[z]].loc[current_dt, 'close'])
            except:
                call_list_prices.append(0)
            try:
                put_list_prices.append(opt[opt['scrip'] == pe_scrip_list[z]].loc[current_dt, 'close'])
            except:
                put_list_prices.append(0)
        call,put,min_diff = call_list_prices[0],put_list_prices[0],float('inf')
        target_2, target_3 = target*2*tf, target*3

        diff = abs(put-call)
        required_call, required_put = None, None
        if (put+call >= target_2) & (min_diff > diff) & (put+call <= target_3):
            min_diff = diff
            required_call, required_put = call, put

        for i in range(1,3):
            if (min_diff > abs(put_list_prices[i] - call)) & (put_list_prices[i]+call >= target_2) & (put_list_prices[i]+call <= target_3):
                min_diff = abs(put_list_prices[i] - call)
                required_call, required_put = call, put_list_prices[i]
            if (min_diff > abs(call_list_prices[i] - put)) & (call_list_prices[i]+put >= target_2) & (call_list_prices[i]+put <= target_3):
                min_diff = abs(call_list_prices[i] - put)
                required_call, required_put = call_list_prices[i], put

        ce_scrip, pe_scrip = ce_scrip_list[call_list_prices.index(required_call)], pe_scrip_list[put_list_prices.index(required_put)]
        ce_price, pe_price = opt[opt['scrip'] == ce_scrip].loc[current_dt, 'close'], opt[opt['scrip'] == pe_scrip].loc[current_dt, 'close']
            

    return ce_scrip,pe_scrip,ce_price,pe_price

            