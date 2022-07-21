# Imports
# Run Registry API
from .rhapi import DEFAULT_URL, RhApi

# Script for getting reference run qualities
from . import ref

def fetch_refs(config, data_run, ref_runs):
    # Handle non-configured subsystems
    if "run_reg" not in config: raise error("Ref suggestions not available for this subsystem")

    folder = "runreg_{}".format(config["run_reg"])
    # Get "runback"
    ref_runs.sort()
    runback = ref_runs[0:ref_runs.index(str(data_run))+1]
    runback.reverse()
    # Only interested in at most 150 runs before data run
    if len(runback) > 151:
        runback = runback[0:151]
    
    dqm = retrieve(folder=folder, ref_runs=runback)
    refs = retrieve(folder=folder, table="runs", ref_runs=dqm)
    return refs

def retrieve(max_run=320008, min_run=316766, folder="runreg_csc", table="datasets", ref_runs=[]):

    api = RhApi(DEFAULT_URL, debug = False)

    # Get column names and name of run column
    col_table = api.table(folder=folder, table=table)["columns"]
    cols=[]
    r_num = ""
    r_num_i = 0
    for col in col_table:
        col_name = str(col["name"])
        if _get_data_col(col_name, table): cols.append(col_name)
        if not r_num and _get_run_col(col_name):
            cols.append(col_name)
            r_num = col_name
            r_num_i = cols.index(col_name)

    if not r_num:
        r_num = cols[0]
    
    # Form query
    c = ",".join("r."+x for x in cols)
    q = "select {0} from {1}.{2} r where r.{3}>=:minrun and r.{3}<=:maxrun order by r.{3}".format(c, folder, table, r_num)
    if ref_runs:
        p = {"maxrun":max(ref_runs), "minrun":min(ref_runs)}
    else:
        p = {"maxrun":str(max_run), "minrun":str(min_run)}
    qid = api.qid(q)

    # Allow for fetch() to update a pre-existing dict
    data = {}
    if type(ref_runs) == dict:
        dqm = ref_runs
        ref_runs = list(ref_runs.keys())
    skipped = 0
    it = 0
    while True:
        runs = []
        raw_data = api.json(q, p)["data"]

        for i in range(0, len(raw_data)):
            run = str(raw_data[i][r_num_i])
            # Only fetch runs relevant to AutoDQM
            if run not in ref_runs: continue
            # Get source of data (Global, Express, or PromptReco)
            if "RDA_NAME" in cols:
                rda_name = raw_data[i][cols.index("RDA_NAME")].lower()
            runs.append(run)
            # Only make a new entry for new runs
            if run not in data:
                data[run] = {}
            for j in range(0, len(raw_data[i])):
                if j == r_num_i: continue
                # Handle <folder>.datasets
                if table != "runs":
                    if "is_good" not in data[run]: pass
                    elif not data[run]["is_good"]: continue 
                    else:
                        # The "BAD" tag gets priority
                        if raw_data[i][j] == "BAD":
                            data[run]["is_good"] = False
                        # Skip "NONSET" tags if marked as "GOOD" elsewhere
                        continue
                    if "RDA_NAME" in data[run]:
                        # Update source
                        data[run]["RDA_NAME"] = rda_name
                    # Update status
                    if raw_data[i][j] == "GOOD": data[run]["is_good"] = True
                    elif raw_data[i][j] == "BAD": data[run]["is_good"] = False
                    else: continue
                # Handle <folder>.runs
                else:
                    data[run][cols[j]] = raw_data[i][j]

            if table != "runs":
                # handle cases where all statuses are NOTSET
                if "is_good" not in data[run]: data[run]["is_good"] = False

        if len(raw_data) < 1 or max(runs) == p["minrun"] or p["minrun"] >= p["maxrun"]:
            break
        p["minrun"] = max(runs) 
        it += 1

    if table == "runs" and ref_runs:
        refs = {"ref_data":[], "ref_cands":[]}
        for run in data:
            if run == max(ref_runs): continue
            refs["ref_data"].append(dict(ref.get_wbm_data(max(ref_runs), run, data), **dqm[run]))
        refs["ref_cands"] = ref.get_ref_cands(refs["ref_data"])
        return refs
    elif data:
        return data 
    else:
        return None

def _get_data_col(col_name, table):
    if table == "datasets":
        return "RDA_CMP" in col_name and "COMMENT" not in col_name and "CAUSE" not in col_name or "RDA_NAME" in col_name
    if table == "runs":
        return "TRIGGERS" in col_name or "LUMI" in col_name or "TIME" in col_name

def _get_run_col(col_name):
    r_num_names = ["runnumber", "run_number"]
    if col_name.lower() in r_num_names:
        return True 
    else:
        for r_name in r_num_names:
            if r_name in col_name.lower():
                return True
    return False

class error(Exception):
    pass
