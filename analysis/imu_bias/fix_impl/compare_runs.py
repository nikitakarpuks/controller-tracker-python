"""Compare fused (pose.csv) and raw-vision (vision_pose.csv) error vs mocap between an OLD run and a NEW run
on their common timestamps (after a warm-up), bridge composed (bridge convention: T_vision.compose(bridge) ~ mocap_rel)."""
import csv, sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python")
from src.mocap_data import load_mocap_bridge
from src.transformations import Transform
REPO = "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python"

def load_poses(path):
    out = {}
    with open(path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            if row[2] in ("", "nan"): continue
            ts, c = int(row[0]), row[1]
            q = list(map(float, row[2:6])); p = list(map(float, row[6:9]))
            out.setdefault(c, {})[ts] = Transform(Rotation.from_quat(q).as_matrix(), np.array(p))
    return out

def load_gt(path):
    g = {}
    with open(path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            ts = int(row[0]); p = list(map(float, row[1:4])); qw, qx, qy, qz = map(float, row[4:8])
            g[ts] = Transform(Rotation.from_quat([qx, qy, qz, qw]).as_matrix(), np.array(p))
    return g

def rot(R): return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

def main(old_dir, new_dir, warmup_frames, label):
    print(f"\n===== {label}")
    for kind, fn in (("FUSED (reported)", "pose.csv"), ("RAW vision", "vision_pose.csv")):
        old, new = load_poses(f"{old_dir}/{fn}"), load_poses(f"{new_dir}/{fn}")
        for c in ("left_controller", "right_controller"):
            b = load_mocap_bridge(f"{REPO}/data/mocap_calib/controller_{c.split('_')[0]}_mocap_bridge_basalt01.json")
            gt = load_gt(f"{new_dir}/mocap_gt/{c}_mocap_gt.csv")
            common = sorted(set(old.get(c, {})) & set(new.get(c, {})) & set(gt))
            common = common[warmup_frames:]
            if len(common) < 20: print(f"  {kind} {c}: only {len(common)} common"); continue
            row = {}
            for name, P in (("old", old), ("new", new)):
                e = [gt[t].inverse().compose(P[c][t].compose(b)) for t in common]
                e = [x.inverse() if False else x for x in e]
                # residual = (T.compose(bridge)).inverse().compose(gt)
                res = [P[c][t].compose(b).inverse().compose(gt[t]) for t in common]
                pe = np.array([np.linalg.norm(r.t) * 1000 for r in res]); re = np.array([rot(r.R) for r in res])
                row[name] = (pe, re)
            for name in ("old", "new"):
                pe, re = row[name]
                print(f"  {kind:17s} {c:16s} {name}: n={len(common)} pos mean {pe.mean():6.2f} med {np.median(pe):6.2f} p90 {np.percentile(pe,90):6.2f} max {pe.max():7.1f} mm | rot mean {re.mean():5.2f} med {np.median(re):5.2f} deg")
            d = row["new"][0] - row["old"][0]
            print(f"      paired (new-old) pos: mean {d.mean():+6.2f} mm; frames better/worse: {(d < -0.5).sum()}/{(d > 0.5).sum()}")

if __name__ == "__main__":
    main(f"{REPO}/visualization/evaluate_2026-09-22/static_dark", f"{REPO}/analysis/imu_bias/fix_impl/e2e_static_dark", 60, "static_dark frames 2900-3500 (first 60 common frames skipped as warm-up)")
