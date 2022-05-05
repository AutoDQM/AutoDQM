from datetime import datetime
import numpy

def get_ref_cands(ref_data):
    """Check various cuts to get 1st and 2nd order reference
    candidates.
    
    Returns a dictionary of ref candidates with relevant info
    of the form: {'run_number':{'info_name':info, 
                                            ..., 
                                    'order':N, 
                                     'best':True/False}}"""

    ref_cands = [] 

    best_ref_i = None
    first_order = {}

    best_lumi_ratio = None
    most_recent = None
    for ref_run in ref_data:
        # Recency
        this_age = ref_run["run_age"]["total"]
        if not most_recent or this_age < most_recent:
            most_recent = this_age
            first_order = ref_run
        # Statistics
        if not ref_run["trigs_cut"]: continue
        # Lumi ratio
        this_lumi_ratio = ref_run["lumi_ratio"]
        if ref_run["lumi_ratio_cut"]:
            if not best_lumi_ratio: pass
            elif abs(1 - this_lumi_ratio) < abs(1 - best_lumi_ratio) and ref_run["run_age"]["days"] < 10:
                best_ref_i = len(ref_cands)
                ref_cands.append(dict({"order":2, "best":False}, **ref_run))
            best_lumi_ratio = this_lumi_ratio

    # Set first order ref
    ref_cands.append(dict({"order":1, "best":False}, **first_order))

    # Set best ref
    if not best_ref_i:
        ref_cands[0]["best"] = True
    else:
        ref_cands[best_ref_i]["best"] = True
    
    return ref_cands 

def get_wbm_data(data_run, this_run, wbm):
    """Return processed data from WBM service"""

    wbm_data = dict.fromkeys(["lumi_ratio", "run_dur", "run_age", "run_trigs", "trigs_cut", "lumi_ratio_cut"])
    wbm_data["run"] = this_run

    # Get run triggers
    this_run_trigs = wbm[this_run]["TRIGGERS"]
    wbm_data["trigs_cut"] = int(this_run_trigs) > 8*10**7 

    # Get run start and stop times
    data_start = wbm[data_run]["STARTTIME"]
    data_stop = wbm[data_run]["STOPTIME"]
    this_start = wbm[this_run]["STARTTIME"]
    this_stop = wbm[this_run]["STOPTIME"]

    data_start_dt = datetime.strptime(data_start, "%Y-%m-%d %H:%M:%S")
    data_stop_dt = datetime.strptime(data_stop, "%Y-%m-%d %H:%M:%S")
    this_start_dt = datetime.strptime(this_start, "%Y-%m-%d %H:%M:%S")
    this_stop_dt = datetime.strptime(this_stop, "%Y-%m-%d %H:%M:%S")

    # Calculate run duration
    wbm_data["run_dur"] = (this_stop_dt - this_start_dt).total_seconds()

    # Calculate time diff between run starts
    delta_dt = data_start_dt - this_start_dt
    # Difference in start times in [days, hours, minutes, total(in seconds)]
    wbm_data["run_age"] = {"days": delta_dt.days,
                           "hours": delta_dt.seconds//3600,
                           "minutes": (delta_dt.seconds//60)%60,
                           "total": delta_dt.total_seconds()}

    # Get luminosity ratio
    try:
        data_init_lumi = float(wbm[data_run]["INITLUMI"])
        data_end_lumi = float(wbm[data_run]["ENDLUMI"])
        data_avg_lumi = _get_avg_lumi(data_init_lumi, data_end_lumi)
        this_init_lumi = float(wbm[this_run]["INITLUMI"])
        this_end_lumi = float(wbm[this_run]["ENDLUMI"])
        this_avg_lumi = _get_avg_lumi(this_init_lumi, this_end_lumi)
    except TypeError: 
        # Handle 'null' entries
        data_avg_lumi = 0
        this_avg_lumi = 0

    if this_avg_lumi*data_avg_lumi != 0:
        wbm_data["lumi_ratio"] = round(data_avg_lumi/this_avg_lumi, 3)
        wbm_data["lumi_ratio_cut"] = abs(1 - wbm_data["lumi_ratio"]) < 0.15
    else:
        wbm_data["lumi_ratio"] = "Not available" 
        wbm_data["lumi_ratio_cut"] = False 



    return wbm_data 

def _get_avg_lumi(init_lumi, end_lumi):
    """Return average run luminosity"""

    if init_lumi*end_lumi == 0: return 0
    if not init_lumi or not end_lumi: return 0

    diff = init_lumi-end_lumi
    quot = init_lumi/end_lumi

    avg = diff/numpy.log(quot)#ROOT.TMath.Log(quot)

    return avg

if __name__ == "__main__":
    pass
