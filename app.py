# -*- coding: utf-8 -*-
import os
import io
import json
import time
import requests
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

st.set_page_config(
    page_title="Cork City 48h Compound Flood Forecaster (M1-v2 Ensemble)",
    page_icon="🌊",
    layout="wide",
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

# ۸ ایستگاه واقعی کریدور رودخانه Lee و بندر کورک (مرتب از غرب به شرق)
OPW_CORRIDOR_STATIONS = [
    ("19102", "Waterworks Weir",          "Upstream Boundary"),
    ("19113", "County Hall",              "Upper Split / South"),
    ("19162", "Fitzgerald's Park",        "Weir Transition"),
    ("19164", "Mercy Hospital",           "Interior (North Ch.)"),
    ("19163", "Pope's Quay",              "City Centre (North)"),
    ("19161", "St Patrick's Quay (East)", "City Centre (East)"),
    ("19160", "Currach Club",             "Lower Tidal Reach"),
    ("19167", "Tivoli Docks",             "Downstream Boundary"),
]

# -------------------------------------------------------------------------
# بارگذاری هسته مدل امولاتور و لایه‌های ثابت (کش دائمی روی سرور)
# -------------------------------------------------------------------------
@st.cache_resource
def load_core_assets():
    pkg = np.load(os.path.join(DATA_DIR, "cork_static_and_bc.npz"))
    dtm         = pkg["dtm"].astype(np.float32)
    manning_n   = pkg["manning_n"].astype(np.float32)
    domain_mask = pkg["domain_mask"].astype(bool)
    river_mask  = pkg["river_mask"].astype(bool)
    gt          = pkg["grid_transform"]
    X_bc_all    = pkg["X_bc_all"].astype(np.float32)
    X_init_all  = pkg["X_init_all"].astype(np.float16)

    ood_cfg = {
        "s_mu": pkg["ood_s_mu"].astype(np.float64),
        "s_cov_inv": pkg["ood_s_cov_inv"].astype(np.float64),
        "thresh": float(pkg["ood_thresh"][0]),
    }

    H, W = dtm.shape
    channel_mask    = domain_mask & (river_mask | (dtm < 2.0))
    floodplain_mask = domain_mask & (~channel_mask)
    extent_km = [gt[2] / 1000.0, (gt[2] + W * gt[0]) / 1000.0,
                 (gt[5] + H * gt[4]) / 1000.0, gt[5] / 1000.0]

    with open(os.path.join(DATA_DIR, "m1_v2_norm_stats.json"), "r") as f:
        norm_stats = json.load(f)
    with open(os.path.join(DATA_DIR, "split_indices_v2.json"), "r") as f:
        split_v2 = json.load(f)

    dtm_norm = np.where(domain_mask, (dtm - norm_stats["dtm_mu"]) / norm_stats["dtm_std"], 0.0).astype(np.float32)
    man_norm = np.where(domain_mask, (manning_n - norm_stats["man_mu"]) / norm_stats["man_std"], 0.0).astype(np.float32)
    x_coord  = np.broadcast_to(np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :], (H, W))
    y_coord  = np.broadcast_to(np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None], (H, W))
    static_6ch = np.stack([
        dtm_norm, man_norm, domain_mask.astype(np.float32),
        channel_mask.astype(np.float32), x_coord * domain_mask, y_coord * domain_mask
    ], axis=0).astype(np.float32)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = min(2, os.cpu_count() or 2)
    sessions = [
        ort.InferenceSession(os.path.join(MODELS_DIR, f"m1_v2_ens_member{i}.onnx"),
                             sess_options=opts, providers=["CPUExecutionProvider"])
        for i in range(3)
    ]

    return {
        "dtm": dtm, "domain_mask": domain_mask, "channel_mask": channel_mask,
        "floodplain_mask": floodplain_mask, "static_6ch": static_6ch,
        "X_bc_all": X_bc_all, "X_init_all": X_init_all, "norm_stats": norm_stats,
        "split_v2": split_v2, "extent_km": extent_km, "ood_cfg": ood_cfg,
        "sessions": sessions,
    }

def load_test_scenario(k_sel):
    with np.load(os.path.join(DATA_DIR, "cork_test10_truth_fp16.npz")) as ft:
        y_true = ft[f"t_{k_sel}"].astype(np.float32)
    with np.load(os.path.join(DATA_DIR, "cork_test10_ens_fp16.npz")) as fe:
        y_mean = fe[f"m_{k_sel}"].astype(np.float32)
        y_std  = fe[f"s_{k_sel}"].astype(np.float32)
    return y_true, y_mean, y_std

# -------------------------------------------------------------------------
# توابع پایش زنده OPW waterlevel.ie و تفکیک هارمونیک خیزاب (کش ۱۵ دقیقه‌ای)
# -------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_opw_latest_table():
    url = "https://waterlevel.ie/geojson/latest/"
    hdr = {"User-Agent": "Mozilla/5.0 (CorkFloodSurrogateDashboard/2.0)"}
    r = requests.get(url, headers=hdr, timeout=20)
    r.raise_for_status()
    feats = r.json().get("features", [])

    ref_meta = {ref: (name, role, idx) for idx, (ref, name, role) in enumerate(OPW_CORRIDOR_STATIONS)}
    found = {}
    for ft in feats:
        pr = ft.get("properties", {})
        ref_raw = str(pr.get("station_ref") or pr.get("ref") or "").lstrip("0")
        sensor  = str(pr.get("sensor_ref") or pr.get("sensor") or "")
        if ref_raw in ref_meta and sensor == "OD":
            coords = (ft.get("geometry") or {}).get("coordinates", [np.nan, np.nan])
            found[ref_raw] = {
                "order": ref_meta[ref_raw][2],
                "Station ID": ref_raw,
                "Station Name": ref_meta[ref_raw][0],
                "Model Role": ref_meta[ref_raw][1],
                "Latest Stage (mOD Malin)": pr.get("value"),
                "Timestamp (UTC)": str(pr.get("datetime") or "").replace("T", " ").replace("Z", ""),
                "Lat": round(float(coords[1]), 5) if len(coords) >= 2 else np.nan,
                "Lon": round(float(coords[0]), 5) if len(coords) >= 2 else np.nan,
                "csv_file": str(pr.get("csv_file") or f"/data/month/{ref_raw}_OD.csv"),
            }
    if not found:
        return pd.DataFrame()
    return pd.DataFrame(found.values()).sort_values("order").drop(columns=["order"]).reset_index(drop=True)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_opw_month_series(ref_id, csv_rel_path=""):
    if csv_rel_path and csv_rel_path.startswith("http"):
        url = csv_rel_path
    elif csv_rel_path and csv_rel_path.startswith("/"):
        url = "https://waterlevel.ie" + csv_rel_path
    else:
        url = f"https://waterlevel.ie/data/month/{ref_id}_OD.csv"

    hdr = {"User-Agent": "Mozilla/5.0 (CorkFloodSurrogateDashboard/2.0)"}
    r = requests.get(url, headers=hdr, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().lower() for c in df.columns]
    dt  = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce")
    val = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s = pd.Series(val.values, index=dt, name=ref_id).dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    # هم‌گام‌سازی ۱۵ دقیقه‌ای برای هم‌ترازی دقیق ایستگاه‌ها و سرعت ترسیم
    return s.resample("15min").mean().interpolate(limit=4)

def harmonic_tide_decompose_8c(series_od):
    """جداسازی مد نجومی و خیزاب توفانی با ۸ مؤلفه اصلی (M2, S2, N2, K2, K1, O1, P1, M4)"""
    s_clean = series_od.dropna()
    periods_hr = np.array([
        12.4206012, 12.0000000, 12.6583482, 11.9672361,
        23.9344696, 25.8193417, 24.0658902,  6.2103006
    ])
    omegas = 2.0 * np.pi / periods_hr
    t_h = (s_clean.index - s_clean.index[0]).total_seconds().values / 3600.0
    y = s_clean.values.astype(np.float64)

    cols = [np.ones_like(t_h)]
    for w in omegas:
        cols.append(np.cos(w * t_h))
        cols.append(np.sin(w * t_h))
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    tide_astro = pd.Series(A @ coef, index=s_clean.index, name="tide_astro")
    surge_res  = pd.Series(y - tide_astro.values, index=s_clean.index, name="storm_surge")
    r2 = float(1.0 - np.var(surge_res) / max(np.var(y), 1e-9))
    return tide_astro, surge_res, r2

# -------------------------------------------------------------------------
# توابع شبیه‌ساز What-If و استنتاج ONNX
# -------------------------------------------------------------------------
def gamma_pulse(t, t_peak, width_hr, skewness=3.0):
    u = np.clip((t - (t_peak - width_hr)) / max(width_hr, 1e-3), 0.0, None)
    return np.where(u > 0, (u ** skewness) * np.exp(skewness * (1.0 - u)), 0.0)

def synthesize_whatif_bc(q_base, q_peak, t_q_peak, q_rise_hr, dbl_ratio,
                         tidal_amp, surge_peak, surge_dur_hr, phase_lag_hr):
    t_hr = np.arange(0, 121, dtype=np.float64)
    main_w = gamma_pulse(t_hr, t_peak=t_q_peak, width_hr=q_rise_hr, skewness=3.2)
    if dbl_ratio > 0.02:
        pre_w = dbl_ratio * gamma_pulse(t_hr, t_peak=t_q_peak - 24.0, width_hr=16.0, skewness=3.0)
        comb = np.clip(main_w + pre_w, 0.0, 1.0)
    else:
        comb = main_w
    q_s = (q_base + (q_peak - q_base) * comb).astype(np.float32)

    t_srg  = t_q_peak + phase_lag_hr
    tide_s = (-0.02 + tidal_amp * np.cos(2.0 * np.pi * (t_hr - t_srg) / 12.5)).astype(np.float32)
    srg_s  = (surge_peak * gamma_pulse(t_hr, t_peak=t_srg, width_hr=surge_dur_hr, skewness=2.4)).astype(np.float32)
    twl_s  = (tide_s + srg_s).astype(np.float32)

    dq   = np.diff(q_s,   prepend=q_s[0]).astype(np.float32)
    dtwl = np.diff(twl_s, prepend=twl_s[0]).astype(np.float32)
    return np.stack([q_s, twl_s, tide_s, srg_s, dq, dtwl], axis=-1)

def compute_ood_distance(bc_121, ood_cfg):
    vec = np.array([
        bc_121[:, 0].max(), bc_121[72:, 0].mean(), bc_121[72:, 1].max(),
        bc_121[72:, 3].max(), bc_121[72, 0],
        float(np.argmax(bc_121[:, 1]) - np.argmax(bc_121[:, 0]))
    ], dtype=np.float64)
    diff = vec - ood_cfg["s_mu"]
    dist = float(np.sqrt(max(diff @ ood_cfg["s_cov_inv"] @ diff, 0.0)))
    return dist, dist > ood_cfg["thresh"]

def estimate_init_depth_analog(bc_121, assets):
    X_all = assets["X_bc_all"]
    q_ref = np.stack([X_all[:, 66, 0] / 300.0, X_all[:, 72, 0] / 300.0,
                      X_all[:, 66, 1] / 2.0,   X_all[:, 72, 1] / 2.0], axis=1)
    q_cur = np.array([bc_121[66, 0] / 300.0, bc_121[72, 0] / 300.0,
                      bc_121[66, 1] / 2.0,   bc_121[72, 1] / 2.0])
    dists = np.linalg.norm(q_ref - q_cur[None, :], axis=1)
    top3  = np.argsort(dists)[:3]
    w = 1.0 / (dists[top3] + 1e-4)
    w /= w.sum()
    return sum(w[j] * assets["X_init_all"][int(top3[j])].astype(np.float32) for j in range(3))

def run_onnx_forecast(bc_121, x_init, assets, step_stride=2, n_members=3, progress_bar=None):
    ns = assets["norm_stats"]
    bc_mu = np.array(ns["bc_mean"], dtype=np.float32)[None, :]
    bc_sd = np.array(ns["bc_std"], dtype=np.float32)[None, :]
    bc_norm = (bc_121 - bc_mu) / bc_sd

    dtm   = assets["dtm"]
    dom_f = assets["domain_mask"].astype(np.float32)
    base_8 = np.concatenate([assets["static_6ch"], np.clip(x_init, 0.0, 12.0) / 5.0], axis=0)

    ks = list(range(0, 48, step_stride))
    if 47 not in ks:
        ks.append(47)
    T_sel = len(ks)
    H, W = dtm.shape

    preds_members = np.zeros((n_members, T_sel, H, W), dtype=np.float32)
    total_calls = n_members * T_sel
    done_calls = 0

    for j, k in enumerate(ks):
        t = 73 + k
        bc_seq  = bc_norm[t - 48 : t + 1, :].T[None, :, :].astype(np.float32)
        twl_now = float(bc_121[t, 1])
        bathtub = (np.clip(twl_now - dtm, 0.0, 12.0) * dom_f / 5.0)[None, None, :, :]
        x_sp    = np.concatenate([base_8[None, :, :, :], bathtub], axis=1).astype(np.float32)
        lead_n  = np.array([[(k + 1) / 48.0]], dtype=np.float32)
        feed    = {"x_sp": x_sp, "bc_seq": bc_seq, "lead_norm": lead_n, "dom_mask": dom_f}

        for m in range(n_members):
            preds_members[m, j] = assets["sessions"][m].run(["depth_out"], feed)[0][0]
            done_calls += 1
            if progress_bar is not None:
                progress_bar.progress(done_calls / total_calls)

    hours_sel = np.array(ks, dtype=np.float32)
    mean_sel  = preds_members.mean(axis=0)
    std_sel   = preds_members.std(axis=0) if n_members > 1 else np.zeros_like(mean_sel)

    if step_stride == 1:
        return mean_sel, std_sel

    mean_48 = np.zeros((48, H, W), dtype=np.float32)
    std_48  = np.zeros((48, H, W), dtype=np.float32)
    for k in range(48):
        pos = np.searchsorted(hours_sel, k)
        if pos == 0:
            mean_48[k], std_48[k] = mean_sel[0], std_sel[0]
        elif pos >= len(hours_sel):
            mean_48[k], std_48[k] = mean_sel[-1], std_sel[-1]
        elif hours_sel[pos] == k:
            mean_48[k], std_48[k] = mean_sel[pos], std_sel[pos]
        else:
            w = (k - hours_sel[pos - 1]) / (hours_sel[pos] - hours_sel[pos - 1])
            mean_48[k] = (1 - w) * mean_sel[pos - 1] + w * mean_sel[pos]
            std_48[k]  = (1 - w) * std_sel[pos - 1]  + w * std_sel[pos]
    return mean_48, std_48

# -------------------------------------------------------------------------
# رابط کاربری اصلی (۴ تب)
# -------------------------------------------------------------------------
assets  = load_core_assets()
fp_mask = assets["floodplain_mask"]
ext     = assets["extent_km"]

st.title("🌊 Cork City 48h Compound Flood Surrogate Dashboard")
st.caption("Emulating HEC-RAS 7.0.2 (20m Sub-grid) via 3-Member Deep Ensemble (2D U-Net + Causal TCN/FiLM + Hurdle Head) | Active Learning v2 (79 Scenarios) + Live OPW Telemetry")

tab_live, tab_bench, tab_eval, tab_opw = st.tabs([
    "⚡ Live What-If Forecaster (ONNX CPU)",
    "🎯 Benchmark Verification (vs. HEC-RAS)",
    "📊 Model Progression & Metrics",
    "📡 Live OPW Telemetry & Surge (River Lee)",
])

# === تب ۱: شبیه‌ساز تعاملی What-If ===
with tab_live:
    c_ctrl, c_view = st.columns([1.05, 2.35])
    with c_ctrl:
        st.subheader("1) Boundary Condition Controls")
        preset = st.selectbox("Quick Storm Preset", [
            "Custom Storm",
            "T200 Extreme Tide (Q=234, Surge=1.05m)",
            "Nov 2009 Fluvial Event (Q=678, Double-Peak)",
            "Severe Compound (Q=750, Spring + 1.05m Surge)",
            "OOD Stress Test (Q=950, Surge=1.35m -> Triggers Alert)",
        ])
        defaults = {
            "Custom Storm": (85, 520, 1.90, 0.90, 0, 0.25),
            "T200 Extreme Tide (Q=234, Surge=1.05m)": (60, 234, 1.95, 1.05, 0, 0.0),
            "Nov 2009 Fluvial Event (Q=678, Double-Peak)": (90, 678, 1.75, 0.69, 0, 0.45),
            "Severe Compound (Q=750, Spring + 1.05m Surge)": (110, 750, 1.95, 1.05, 0, 0.35),
            "OOD Stress Test (Q=950, Surge=1.35m -> Triggers Alert)": (130, 950, 1.95, 1.35, 0, 0.40),
        }[preset]

        q_peak = st.slider("Upstream Peak Flow Q_peak (m³/s) [Waterworks]", 140, 1000, int(defaults[1]), 10)
        q_base = st.slider("Initial Baseflow Q_base (m³/s)", 35, 150, int(defaults[0]), 5)
        t_amp  = st.slider("Astronomical Tide Amplitude (m) [Neap 1.03 .. Spring 1.95]", 1.03, 1.95, float(defaults[2]), 0.02)
        srg_pk = st.slider("Storm Surge Peak at Tivoli (m)", 0.05, 1.45, float(defaults[3]), 0.05)
        lag_hr = st.slider("Peak Lag: TWL minus Q (hours)", -10, 10, int(defaults[4]), 1)
        dbl_r  = st.slider("Pre-storm Double-Peak Ratio (2009-style)", 0.0, 0.55, float(defaults[5]), 0.05)

        mode_cpu = st.radio("Inference Mode (Cloud CPU)", [
            "Fast Ensemble (2h stride, 3 members)",
            "Full Ensemble (1h stride, 3 members)",
            "Ultra-Fast Single Member (2h stride, 1 member)",
        ], index=0)

        bc_live = synthesize_whatif_bc(q_base, q_peak, 90.0, 24.0, dbl_r, t_amp, srg_pk, 16.0, lag_hr)
        ood_d, ood_flag = compute_ood_distance(bc_live, assets["ood_cfg"])

        st.markdown("---")
        if ood_flag:
            st.error(f"🚨 **OOD ALERT (Mahalanobis = {ood_d:.2f} > {assets['ood_cfg']['thresh']:.2f})**\n\n"
                     "Event outside the 60-scenario training envelope — **trigger HEC-RAS 7.0.2 fallback**.")
        else:
            st.success(f"✅ **In-Distribution (Mahalanobis = {ood_d:.2f} ≤ {assets['ood_cfg']['thresh']:.2f})**\n\n"
                       "Safe operating regime for surrogate forecasting.")

        run_btn = st.button("🚀 Run 48h Surrogate Forecast", type="primary", use_container_width=True)

    with c_view:
        fig_bc, ax1 = plt.subplots(figsize=(9.5, 2.7))
        hrs_fcst = np.arange(-24, 49)
        ax1.plot(hrs_fcst, bc_live[48:121, 0], color="navy", lw=2.0)
        ax1.axhline(350, color="navy", ls=":", lw=1, alpha=0.6)
        ax1.axvline(0, color="black", ls="--", lw=1.2)
        ax1.set_ylabel("Discharge Q (m³/s)", color="navy")
        ax1.set_xlabel("Hours Relative to Forecast Start t0 (0h = Now, +48h = Horizon)")
        ax2 = ax1.twinx()
        ax2.plot(hrs_fcst, bc_live[48:121, 1], color="crimson", lw=1.8)
        ax2.axhline(2.50, color="crimson", ls=":", lw=1, alpha=0.6)
        ax2.set_ylabel("Tivoli TWL (mOD Malin)", color="crimson")
        ax1.set_title(f"Input Forcing | Q_peak = {bc_live[:,0].max():.0f} m³/s | Max TWL = {bc_live[72:,1].max():.2f} mOD")
        ax1.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig_bc)
        plt.close(fig_bc)

        if run_btn:
            stride = 1 if "1h stride" in mode_cpu else 2
            n_mem  = 1 if "Single Member" in mode_cpu else 3
            pbar = st.progress(0.0, text="Running ONNX Runtime ensemble inference...")
            t_s = time.time()
            x_in_est = estimate_init_depth_analog(bc_live, assets)
            m_48, s_48 = run_onnx_forecast(bc_live, x_in_est, assets, step_stride=stride, n_members=n_mem, progress_bar=pbar)
            elapsed_s = time.time() - t_s
            pbar.empty()
            st.session_state["live_res"] = {
                "mean_48": m_48, "std_48": s_48, "elapsed_s": elapsed_s, "n_mem": n_mem
            }

        if "live_res" in st.session_state:
            res = st.session_state["live_res"]
            m_48, s_48 = res["mean_48"], res["std_48"]
            peak_m = m_48.max(axis=0)
            area_05 = float(np.sum((peak_m > 0.05) & fp_mask) * 400.0 / 1e6)
            area_30 = float(np.sum((peak_m > 0.30) & fp_mask) * 400.0 / 1e6)
            max_fp_d = float(peak_m[fp_mask].max())
            pk_hr = int(np.argmax(m_48[:, fp_mask].mean(axis=1))) + 1

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Flooded Urban Area (>5cm)", f"{area_05:.2f} km²")
            k2.metric("Severe Flood (>30cm)", f"{area_30:.2f} km²")
            k3.metric("Max Floodplain Depth", f"{max_fp_d:.2f} m", f"Peak at t0+{pk_hr}h")
            k4.metric("CPU Inference Time", f"{res['elapsed_s']:.1f} s", "vs 50s–540s HEC-RAS")

            view_opt = st.radio("Map Layer", ["Peak 48h Envelope", "Hourly Snapshot (Slider)"], horizontal=True)
            if view_opt == "Hourly Snapshot (Slider)":
                hr_sel = st.slider("Forecast Lead Hour (t0 + h)", 1, 48, pk_hr)
                d_show, u_show, lbl_t = m_48[hr_sel - 1], s_48[hr_sel - 1] * 100.0, f"t0 + {hr_sel}h"
            else:
                d_show, u_show, lbl_t = peak_m, s_48[pk_hr - 1] * 100.0, "48h Maximum Envelope"

            fig_m, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))
            axes[0].imshow(assets["dtm"], extent=ext, cmap="gray", vmin=0, vmax=35, alpha=0.4)
            im0 = axes[0].imshow(np.ma.masked_where(~fp_mask | (d_show < 0.05), d_show),
                                 extent=ext, cmap="YlGnBu", vmin=0.05, vmax=2.5)
            axes[0].set_title(f"Predicted Floodplain Depth ({lbl_t})")
            axes[0].set_xlabel("ITM Easting (km)"); axes[0].set_ylabel("ITM Northing (km)")
            plt.colorbar(im0, ax=axes[0], label="Depth (m)", fraction=0.03)

            axes[1].imshow(assets["dtm"], extent=ext, cmap="gray", vmin=0, vmax=35, alpha=0.4)
            im1 = axes[1].imshow(np.ma.masked_where(~fp_mask | (d_show < 0.05), u_show),
                                 extent=ext, cmap="magma", vmin=0, vmax=15)
            axes[1].set_title(f"Epistemic Uncertainty Std ({res['n_mem']}-Member Ensemble)")
            axes[1].set_xlabel("ITM Easting (km)")
            plt.colorbar(im1, ax=axes[1], label="Std (cm)", fraction=0.03)
            plt.tight_layout()
            st.pyplot(fig_m)
            plt.close(fig_m)
        else:
            st.info("Select a storm preset or adjust sliders on the left, then click **Run 48h Surrogate Forecast**.")

# === تب ۲: مقایسه با ۱۰ سناریوی آزمون HEC-RAS ===
with tab_bench:
    st.subheader("Instant Verification on Frozen 10-Scenario Test Set (Zero Latency)")
    df_ens   = pd.read_csv(os.path.join(DATA_DIR, "m1_v2_ensemble_test_metrics.csv"))
    test_ids = assets["split_v2"]["test_ids"]

    c_sel, c_hr = st.columns([1.2, 1.8])
    with c_sel:
        k_sel = st.selectbox("Select Unseen Test Scenario", range(len(test_ids)),
                             format_func=lambda i: f"{test_ids[i]}  (MAE={df_ens.loc[i,'MAE_wet_cm']:.2f} cm | PeakMAE={df_ens.loc[i,'Peak_MAE_cm']:.2f} cm)")
    with c_hr:
        mode_b = st.radio("View Mode", ["48h Peak Map", "Hourly Map"], horizontal=True)

    y_true_48, y_mean_48, y_std_48 = load_test_scenario(k_sel)
    pk_h_b = int(np.argmax(y_true_48[:, fp_mask].mean(axis=1))) + 1

    if mode_b == "Hourly Map":
        h_b = st.slider("Test Scenario Hour (t0 + h)", 1, 48, pk_h_b, key="bench_hr")
        tp_m, pp_m, ttl = y_true_48[h_b - 1], y_mean_48[h_b - 1], f"t0+{h_b}h"
    else:
        tp_m, pp_m, ttl = y_true_48.max(axis=0), y_mean_48.max(axis=0), "48h Peak"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Wet MAE", f"{df_ens.loc[k_sel,'MAE_wet_cm']:.2f} cm")
    m2.metric("Peak MAE", f"{df_ens.loc[k_sel,'Peak_MAE_cm']:.2f} cm")
    m3.metric("CSI @ 5cm", f"{df_ens.loc[k_sel,'CSI_05cm']:.3f}")
    m4.metric("CSI @ 30cm", f"{df_ens.loc[k_sel,'CSI_30cm']:.3f}")
    m5.metric("90% Coverage", f"{df_ens.loc[k_sel,'Cov90_pct']:.1f}%")

    fig_b, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    im0 = axes[0].imshow(np.ma.masked_where(~fp_mask | (tp_m < 0.05), tp_m), extent=ext, cmap="YlGnBu", vmin=0.05, vmax=2.5)
    axes[0].set_title(f"HEC-RAS 7.0.2 Truth ({ttl})")
    plt.colorbar(im0, ax=axes[0], fraction=0.03, label="m")

    im1 = axes[1].imshow(np.ma.masked_where(~fp_mask | (pp_m < 0.05), pp_m), extent=ext, cmap="YlGnBu", vmin=0.05, vmax=2.5)
    axes[1].set_title(f"M1-v2 Ensemble Mean ({ttl})")
    plt.colorbar(im1, ax=axes[1], fraction=0.03, label="m")

    err_cm = (pp_m - tp_m) * 100.0
    im2 = axes[2].imshow(np.ma.masked_where(~fp_mask | ((tp_m < 0.05) & (pp_m < 0.05)), err_cm),
                         extent=ext, cmap="RdBu_r", vmin=-15, vmax=15)
    axes[2].set_title("Error: Surrogate - Truth (cm)")
    plt.colorbar(im2, ax=axes[2], fraction=0.03, label="cm")
    plt.tight_layout()
    st.pyplot(fig_b)
    plt.close(fig_b)

    fp_coords = np.argwhere(fp_mask)
    pk_vals_fp = y_true_48.max(axis=0)[fp_mask]
    p90_idx = np.argsort(pk_vals_fp)[int(0.90 * len(pk_vals_fp))]
    ry, rx = fp_coords[p90_idx]

    hrs = np.arange(1, 49)
    tr_ts, mn_ts, sd_ts = y_true_48[:, ry, rx], y_mean_48[:, ry, rx], y_std_48[:, ry, rx]
    fig_ts, ax_ts = plt.subplots(figsize=(12, 2.8))
    ax_ts.fill_between(hrs, np.maximum(mn_ts - 1.645 * sd_ts, 0.0), mn_ts + 1.645 * sd_ts,
                       color="dodgerblue", alpha=0.25, label="Ensemble P10–P90 Band")
    ax_ts.plot(hrs, tr_ts, "k-", lw=2.0, label="HEC-RAS 7.0.2 Ground Truth")
    ax_ts.plot(hrs, mn_ts, "--", color="royalblue", lw=2.0, label="M1-v2 Ensemble Mean")
    ax_ts.set_xlabel("Forecast Lead Hour (t0 + h)"); ax_ts.set_ylabel("Water Depth (m)")
    ax_ts.set_title(f"48h Probabilistic Hydrograph at Urban Flood Hotspot (Row={ry}, Col={rx})")
    ax_ts.legend(loc="upper right"); ax_ts.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_ts)
    plt.close(fig_ts)

# === تب ۳: جدول پیشرفت مدل‌ها ===
with tab_eval:
    st.subheader("Controlled Progression on Frozen 10-Scenario Test Set")
    prog_df = pd.DataFrame({
        "Model Stage": [
            "M0_v1: Causal POD + Ridge (48 Train)",
            "M0_v2: Causal POD + Ridge (60 Train + AL)",
            "M1_v1: 2D U-Net + FiLM + Hurdle (48 Train)",
            "M1_v2: 3-Member Deep Ensemble + Peak Weighting (60 Train + AL)",
        ],
        "MAE_wet (cm)": [5.98, 5.76, 3.80, 1.94],
        "RMSE_wet (cm)": [9.21, 8.60, 6.67, 2.94],
        "Peak_MAE (cm)": [6.62, 5.69, 5.86, 1.90],
        "CSI @ 5cm": [0.829, 0.844, 0.926, 0.969],
        "CSI @ 30cm": [0.909, 0.915, 0.936, 0.969],
    })
    st.dataframe(prog_df, use_container_width=True, hide_index=True)

    st.subheader("Per-Scenario Test Breakdown (M1-v2 3-Member Ensemble)")
    df_ens = pd.read_csv(os.path.join(DATA_DIR, "m1_v2_ensemble_test_metrics.csv"))
    st.dataframe(df_ens, use_container_width=True, hide_index=True)

# === تب ۴: پایش زنده ۸ ایستگاه واقعی OPW و تفکیک هارمونیک خیزاب ===
with tab_opw:
    hdr_l, hdr_r = st.columns([3.2, 1.0])
    with hdr_l:
        st.subheader("Live OPW Hydrometric Corridor (Waterworks Weir → Tivoli Docks)")
        st.caption("Real-time 15-minute telemetry from OPW waterlevel.ie (OD Malin datum) with 8-constituent harmonic surge decomposition.")
    with hdr_r:
        if st.button("🔄 Refresh OPW Feed", use_container_width=True):
            fetch_opw_latest_table.clear()
            fetch_opw_month_series.clear()
            st.rerun()

    try:
        with st.spinner("Connecting to OPW waterlevel.ie live telemetry..."):
            df_latest = fetch_opw_latest_table()

        if df_latest.empty:
            st.warning("Connected to waterlevel.ie, but no OD Malin readings were returned for the 8 Cork corridor gauges.")
        else:
            # انتخاب پنجره زمانی نمایش (تحلیل هارمونیک همیشه روی کل ۳۰ روز انجام می‌شود)
            win_days = st.radio("Display Window (Harmonic Fit Always Uses Full 30-Day Record)",
                                [3, 7, 14, 30], index=1, horizontal=True, format_func=lambda d: f"Last {d} Days")

            csv_map = dict(zip(df_latest["Station ID"], df_latest["csv_file"]))
            core_refs = ["19102", "19162", "19164", "19161", "19160", "19167"]
            ts_map = {}
            with st.spinner("Downloading 30-day OD Malin time series & running harmonic surge decomposition..."):
                for r_id in core_refs:
                    try:
                        ts_map[r_id] = fetch_opw_month_series(r_id, csv_map.get(r_id, ""))
                    except Exception:
                        pass

            # ساخت سری جزرومدی ترمیم‌شده برای تیولی (رفع بریدگی کف سنسور <= -0.92 mOD با کمک Currach Club)
            tidal_src_label = ""
            n_repaired_hrs  = 0.0
            s_twl_full = None

            if "19167" in ts_map and "19160" in ts_map:
                df_pair = pd.concat({"tiv": ts_map["19167"], "cur": ts_map["19160"]}, axis=1).dropna(how="all")
                s_tiv = df_pair["tiv"]
                s_cur = df_pair["cur"]
                clipped = (s_tiv <= -0.92) | s_tiv.isna()
                valid_both = (~clipped) & s_tiv.notna() & s_cur.notna()
                off_m = float(np.median(s_tiv[valid_both] - s_cur[valid_both])) if valid_both.any() else 0.0
                s_twl_full = s_tiv.where(~clipped, s_cur + off_m).dropna()
                n_repaired_hrs = float(clipped.sum() * 0.25)
                tidal_src_label = f"19167 Tivoli Docks (low-tide sensor floor repaired via 19160 Currach Club, offset={off_m:+.3f} m)"
            elif "19160" in ts_map:
                s_twl_full = ts_map["19160"]
                tidal_src_label = "19160 Currach Club (Lower Tidal Reach)"
            elif "19167" in ts_map:
                s_twl_full = ts_map["19167"]
                tidal_src_label = "19167 Tivoli Docks (Raw)"

            # محاسبه تفکیک هارمونیک ۸ مؤلفه روی کل ۳۰ روز و برش پنجره نمایش
            if s_twl_full is not None and len(s_twl_full) > 96:
                s_tide_full, s_surge_full, r2_tide = harmonic_tide_decompose_8c(s_twl_full)
                t_max_all = s_twl_full.index.max()
                t_cut = t_max_all - pd.Timedelta(days=int(win_days))

                s_twl_w   = s_twl_full.loc[t_cut:]
                s_tide_w  = s_tide_full.loc[t_cut:]
                s_surge_w = s_surge_full.loc[t_cut:]

                ww_val = df_latest.loc[df_latest["Station ID"] == "19102", "Latest Stage (mOD Malin)"]
                ww_now = f"{float(ww_val.iloc[0]):.2f} mOD" if len(ww_val) > 0 and pd.notna(ww_val.iloc[0]) else "N/A"
                twl_now = f"{float(s_twl_w.iloc[-1]):.2f} mOD"
                srg_now = float(s_surge_w.iloc[-1])
                srg_pk  = float(s_surge_w.max())
                srg_pk_t = s_surge_w.idxmax().strftime("%b %d %H:%M UTC")

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Upstream: Waterworks (19102)", ww_now)
                k2.metric("Harbour TWL (Latest)", twl_now, f"Max {win_days}d: {s_twl_w.max():.2f} mOD")
                k3.metric("Current Surge Residual", f"{srg_now:+.2f} m", f"Harmonic R² = {r2_tide:.3f}")
                k4.metric(f"Peak Surge (Last {win_days}d)", f"{srg_pk:+.2f} m", srg_pk_t)

                # نمودار ۳ پنلی زنده (بالادست + مد و مد نجومی + خیزاب استخراج‌شده)
                fig_opw, axes_o = plt.subplots(3, 1, figsize=(13.5, 7.6), sharex=True)

                # پنل ۱: تراز رودخانه بالادست و انتقال شهری
                if "19102" in ts_map:
                    s_ww = ts_map["19102"].loc[t_cut:]
                    axes_o[0].plot(s_ww.index, s_ww.values, color="#1e3a8a", lw=1.8, label="19102 Waterworks Weir (Upstream)")
                if "19162" in ts_map:
                    s_fp = ts_map["19162"].loc[t_cut:]
                    axes_o[0].plot(s_fp.index, s_fp.values, color="#7c3aed", lw=1.4, label="19162 Fitzgerald's Park (Weir Transition)")
                if "19164" in ts_map:
                    s_mh = ts_map["19164"].loc[t_cut:]
                    axes_o[0].plot(s_mh.index, s_mh.values, color="#059669", lw=1.1, alpha=0.8, label="19164 Mercy Hospital (Interior)")
                axes_o[0].set_ylabel("Stage (mOD)")
                axes_o[0].set_title(f"Live River Lee Corridor & Cork Harbour Surge Decomposition (Last {win_days} Days, UTC)")
                axes_o[0].legend(loc="upper right", ncol=3, fontsize=8.2)
                axes_o[0].grid(True, alpha=0.3)

                # پنل ۲: تراز کل بندر کورک در برابر مد نجومی ۸ مؤلفه
                axes_o[1].plot(s_twl_w.index, s_twl_w.values, color="crimson", lw=1.4, label="Observed TWL (Repaired)")
                axes_o[1].plot(s_tide_w.index, s_tide_w.values, color="#2563eb", lw=1.0, ls="--", alpha=0.85,
                               label="Astronomical Tide (8 Constituents: M2,S2,N2,K2,K1,O1,P1,M4)")
                axes_o[1].axhline(2.50, color="crimson", ls=":", lw=0.9, alpha=0.6, label="Cork Flood Alert Ref (~2.5 mOD)")
                axes_o[1].set_ylabel("TWL (mOD Malin)")
                axes_o[1].legend(loc="upper right", fontsize=8.0)
                axes_o[1].grid(True, alpha=0.3)

                # پنل ۳: خیزاب توفانی غیرجزرومدی
                axes_o[2].fill_between(s_surge_w.index, 0, s_surge_w.values, where=s_surge_w.values >= 0,
                                       color="#ef4444", alpha=0.45, label="Positive Storm Surge")
                axes_o[2].fill_between(s_surge_w.index, 0, s_surge_w.values, where=s_surge_w.values < 0,
                                       color="#3b82f6", alpha=0.35, label="Negative Surge (High Pressure)")
                axes_o[2].axhline(0.0, color="black", lw=0.8)
                axes_o[2].scatter([s_surge_w.idxmax()], [srg_pk], color="#991b1b", s=35, zorder=5)
                axes_o[2].annotate(f"Peak {srg_pk:+.2f} m ({srg_pk_t})",
                                   xy=(s_surge_w.idxmax(), srg_pk), xytext=(8, 4),
                                   textcoords="offset points", fontsize=8.2, fontweight="bold", color="#991b1b")
                axes_o[2].set_ylabel("Surge Residual (m)")
                axes_o[2].set_xlabel("Date / Time (UTC)")
                axes_o[2].legend(loc="upper right", fontsize=8.2)
                axes_o[2].grid(True, alpha=0.3)
                axes_o[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

                plt.tight_layout()
                st.pyplot(fig_opw)
                plt.close(fig_opw)

                if n_repaired_hrs > 0:
                    st.caption(f"QA/QC Note: {tidal_src_label}. Across the 30-day archive, {n_repaired_hrs:.1f} h of low-tide sensor clipping (≤ -0.92 mOD) were seamlessly patched.")
            else:
                st.info("Tidal time series temporarily unavailable for harmonic decomposition; displaying latest snapshots below.")

            st.markdown("#### Current Snapshot — All 8 Corridor Stations (West → East)")
            show_cols = ["Station ID", "Station Name", "Model Role", "Latest Stage (mOD Malin)", "Timestamp (UTC)", "Lat", "Lon"]
            st.dataframe(df_latest[show_cols], use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"⚠️ Could not reach OPW waterlevel.ie right now ({type(e).__name__}: {e}). Tabs 1–3 continue operating offline normally.")
