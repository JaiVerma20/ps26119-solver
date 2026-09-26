import csv, os, subprocess, sys, json, time
R = "/Users/jai/Desktop/SIH26119"; SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R + "/tools")
from lpm import fingerprint, read_mps_highspy
tl = float(sys.argv[1]) if len(sys.argv) > 1 else 60
opt = {r["name"]: r for r in csv.DictReader(open(R + "/data/netlib/optima.csv"))}
out = open(SP + "/netlib_simplex_audit.csv", "w", newline=""); w = csv.writer(out)
w.writerow(["instance","rows","cols","nnz","fp_teammate","fp_highspy","fp_match","status","iterations","seconds","objective","highs_objective","rel_err_highs","checker","verify","verify_primal","verify_dual","verify_gap","message"])
for name in sorted(opt):
    mps = f"{R}/data/netlib/{name}.mps"; sol = f"{SP}/sol/{name}.sol"; os.makedirs(SP + "/sol", exist_ok=True)
    t0 = time.time()
    try:
        p = subprocess.run([SP + "/audit_driver", mps, sol, str(tl)], capture_output=True, text=True, timeout=tl + 60)
        o = dict(l.split(" ", 1) for l in p.stdout.splitlines() if " " in l)
    except subprocess.TimeoutExpired:
        o = {"status": "HARD_TIMEOUT"}
    try: fph = fingerprint(read_mps_highspy(mps))
    except Exception as e: fph = "ERR"
    ver = {}
    if o.get("status") == "OPTIMAL":
        v = subprocess.run([sys.executable, R + "/tools/verify.py", mps, sol, "--json", SP + "/v.json", "--quiet"], capture_output=True, text=True)
        try: ver = json.load(open(SP + "/v.json"))
        except Exception: ver = {}
    ho = float(opt[name]["highs_objective"]); obj = o.get("objective", "nan")
    try: rel = abs(float(obj) - ho) / (1 + abs(ho))
    except: rel = float("nan")
    row = [name, o.get("rows"), o.get("cols"), o.get("nnz"), o.get("fingerprint"), fph, o.get("fingerprint") == fph, o.get("status"), o.get("iterations"), o.get("seconds"), obj, ho, "%.3g" % rel,
           "PASS" if "PASS" in o.get("checker", "") else ("FAIL" if "FAIL" in o.get("checker", "") else "n/a"),
           ver.get("verdict", "n/a"), ver.get("primal_rel"), ver.get("dual_rel"), ver.get("gap_rel"), o.get("message", "")[:200]]
    w.writerow(row); out.flush(); print(*row[:15], flush=True)
