"""Build gyro interval terms and accel window normal equations for, per (recording, controller):
   stream 'csv'  : the raw CSV values (what the recorder wrote, no correction at all)
   stream 'base' : the project's loader correction (entry 1: M1 @ csv + b1)   [validation vs the oracle]
   stream 'e0'   : entry-0 loader correction (M0 @ csv + b0)                  [direct check of the analytic mapping]
   stream 'sens' : sensor-frame re-application  D (M1 (D csv) + b1)           [direct check, physically-consistent variant]
"""
import sys, pickle, time
from multiprocessing import Pool
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/factory_audit")
from fa_common import *

def streams_for(t, gc, ac, ctrl):
    Mg1, bg1, Ma1, ba1 = factory(ctrl, 1); Mg0, bg0, Ma0, ba0 = factory(ctrl, 0)
    return {
        "csv":  (gc, ac),
        "base": (apply(gc, Mg1, bg1), apply(ac, Ma1, ba1)),
        "e0":   (apply(gc, Mg0, bg0), apply(ac, Ma0, ba0)),
        "sens": ((D @ apply((D @ gc.T).T, Mg1, bg1).T).T, (D @ apply((D @ ac.T).T, Ma1, ba1).T).T),
    }

def job(args):
    name, ctrl, which = args
    t0 = time.time()
    rdir = rec_dir(name)
    t, gc, ac = raw_csv(rdir, ctrl)
    S = streams_for(t, gc, ac, ctrl)
    out = {}
    for sname, (gs, fs) in S.items():
        if which == "gyro":
            out[sname] = gyro_terms_stream(name, ctrl, t, gs)
        else:
            out[sname] = accel_windows_stream(name, ctrl, t, fs)
    pickle.dump(out, open(FA + f"terms_{which}_{name}_{ctrl}.pkl", "wb"))
    return f"{which} {name}/{ctrl} done in {time.time()-t0:.0f}s"

if __name__ == "__main__":
    which = sys.argv[1]
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else ALL8
    jobs = [(n, c, which) for n in names for c in CTRLS]
    with Pool(6) as p:
        for r in p.imap_unordered(job, jobs):
            print(r, flush=True)
