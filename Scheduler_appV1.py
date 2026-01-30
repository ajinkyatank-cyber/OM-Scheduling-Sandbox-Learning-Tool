# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 12:45:59 2026

@author: HP
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="OM Scheduling Sandbox", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def compute_schedule(df: pd.DataFrame, sequence: list[str], m: int):
    """
    Flow-shop style for m machines:
    - For m=1: single machine sequencing.
    - For m=2: same job order on both machines; M2 start = max(M1 completion of job, M2 completion of prev job).
    Returns:
      times: dict[(job, machine)] = (start, finish)
      completion_last: dict[job] = completion time on last machine
    """
    p = {row["Job"]: [row[f"M{k}"] for k in range(1, m + 1)] for _, row in df.iterrows()}

    times = {}
    machine_ready = [0.0] * m
    job_ready_prev_machine = {job: 0.0 for job in sequence}

    for job in sequence:
        for k in range(m):
            start = max(machine_ready[k], job_ready_prev_machine[job])
            finish = start + float(p[job][k])

            times[(job, k + 1)] = (start, finish)
            machine_ready[k] = finish
            job_ready_prev_machine[job] = finish  # completion on this machine

    completion_last = {job: times[(job, m)][1] for job in sequence}
    return times, completion_last


def kpis(df: pd.DataFrame, completion_last: dict[str, float], sequence: list[str]):
    """
    Assumptions for this teaching sandbox:
    - All jobs are available at time 0 (no release dates).
    - Flow time of job j = completion time Cj (on the last machine).
    - Due date is compared to completion on the last machine.
    """
    d = {row["Job"]: float(row["Due"]) for _, row in df.iterrows()}

    rows = []
    for job in sequence:
        C = float(completion_last[job])        # completion on last machine
        due = float(d[job])

        # Flow time (since all jobs "arrive" at time 0)
        F = C

        # Lateness / tardiness / earliness
        L = C - due
        T = max(0.0, L)
        E = max(0.0, -L)

        rows.append({
            "Job": job,
            "Due": due,
            "Completion (Cj)": C,
            "Flow time (Fj)": F,
            "Lateness (C-D)": L,
            "Tardiness (Tj)": T,
            "Earliness (Ej)": E,
        })

    out = pd.DataFrame(rows)

    n = len(out)
    if n == 0:
        return out, {}

    Cmax = float(out["Completion (Cj)"].max())
    total_flow = float(out["Flow time (Fj)"].sum())
    avg_flow = total_flow / n

    total_tardiness = float(out["Tardiness (Tj)"].sum())
    avg_tardiness = total_tardiness / n

    total_earliness = float(out["Earliness (Ej)"].sum())
    avg_earliness = total_earliness / n

    max_lateness = float(out["Lateness (C-D)"].max())
    min_lateness = float(out["Lateness (C-D)"].min())

    num_late = int((out["Tardiness (Tj)"] > 0).sum())
    num_early = int((out["Earliness (Ej)"] > 0).sum())
    num_ontime = n - num_late - num_early  # exactly on time (L=0)

    summary = {
        "Makespan (Cmax)": Cmax,
        "Total flow time (ΣF)": total_flow,
        "Average flow time (F̄)": avg_flow,
        "Total tardiness (ΣT)": total_tardiness,
        "Average tardiness (T̄)": avg_tardiness,
        "Total earliness (ΣE)": total_earliness,
        "Average earliness (Ē)": avg_earliness,
        "# Late jobs": num_late,
        "# Early jobs": num_early,
        "# On-time jobs": num_ontime,
        "Max lateness (Lmax)": max_lateness,
        "Min lateness (Lmin)": min_lateness,
    }
    return out, summary


def plot_gantt(times: dict, sequence: list[str], m: int):
    fig, ax = plt.subplots(figsize=(12, 1.2 + 0.8 * m))

    y_gap = 8
    bar_h = 6
    yticks, ylabels = [], []

    for machine in range(1, m + 1):
        y = (m - machine) * (bar_h + y_gap)
        yticks.append(y + bar_h / 2)
        ylabels.append(f"Machine {machine}")

        for job in sequence:
            start, finish = times[(job, machine)]
            ax.broken_barh([(start, finish - start)], (y, bar_h))
            ax.text(start + (finish - start) / 2, y + bar_h / 2, job,
                    va="center", ha="center", fontsize=9)

    ax.set_xlabel("Time")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.set_title("Gantt Chart (same sequence on all machines)")
    fig.tight_layout()
    return fig


# ---------------------------
# UI
# ---------------------------
st.title("OM Scheduling Sandbox (Exploration)")

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Problem setup")

    m = st.selectbox("Number of machines", [1, 2], index=0)
    n = st.number_input("Number of jobs", min_value=2, max_value=25, value=5, step=1)

    st.caption("Enter processing times and due dates. No release dates / precedence.")

    default_jobs = [f"J{i+1}" for i in range(int(n))]
    cols = ["Job"] + [f"M{k}" for k in range(1, m + 1)] + ["Due"]

    if "data" not in st.session_state or st.session_state.get("m_prev") != m or st.session_state.get("n_prev") != n:
        data = []
        for idx, j in enumerate(default_jobs):
            row = {"Job": j}
            for k in range(1, m + 1):
                row[f"M{k}"] = 1.0
            row["Due"] = float(4 + idx)
            data.append(row)
        st.session_state.data = pd.DataFrame(data, columns=cols)
        st.session_state.m_prev = m
        st.session_state.n_prev = n
        st.session_state.sequence = []

    df = st.data_editor(
        st.session_state.data,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        key="data_editor",
    )

    jobs = df["Job"].astype(str).tolist()
    if df["Job"].duplicated().any():
        st.error("Job names must be unique.")
    if (df[[c for c in df.columns if c.startswith("M")]] <= 0).any().any():
        st.error("All processing times must be > 0.")
    if (df["Due"] <= 0).any():
        st.error("All due dates must be > 0.")

with right:
    st.subheader("2) Build the sequence (student chooses)")

    if "sequence" not in st.session_state:
        st.session_state.sequence = []

    remaining = [j for j in jobs if j not in st.session_state.sequence]

    c1, c2 = st.columns([3, 2])
    with c1:
        pick = st.selectbox("Add next job", remaining if remaining else ["(all scheduled)"])
    with c2:
        if st.button("Add ➕", disabled=(not remaining or pick == "(all scheduled)")):
            st.session_state.sequence.append(pick)

    st.markdown("**Current sequence**")
    seq = st.session_state.sequence

    if not seq:
        st.info("Add jobs to create a sequence.")
    else:
        for i, job in enumerate(seq):
            r1, r2, r3, r4 = st.columns([6, 1, 1, 1])
            r1.write(f"{i+1}. **{job}**")
            if r2.button("⬆️", key=f"up_{i}", disabled=(i == 0)):
                seq[i-1], seq[i] = seq[i], seq[i-1]
                st.session_state.sequence = seq
                st.rerun()
            if r3.button("⬇️", key=f"down_{i}", disabled=(i == len(seq)-1)):
                seq[i+1], seq[i] = seq[i], seq[i+1]
                st.session_state.sequence = seq
                st.rerun()
            if r4.button("❌", key=f"del_{i}"):
                seq.pop(i)
                st.session_state.sequence = seq
                st.rerun()

        c3, c4 = st.columns(2)
        with c3:
            if st.button("Clear sequence 🧹"):
                st.session_state.sequence = []
                st.rerun()
        with c4:
            if st.button("Auto-fill remaining (append)"):
                st.session_state.sequence = seq + remaining
                st.rerun()

st.divider()
st.subheader("3) Results")

if len(st.session_state.sequence) != len(jobs):
    st.warning("Schedule is incomplete. Add all jobs to compute KPIs and Gantt.")
else:
    df2 = df.copy()
    df2["Job"] = df2["Job"].astype(str)

    times, completion_last = compute_schedule(df2, st.session_state.sequence, m)
    per_job, summary = kpis(df2, completion_last, st.session_state.sequence)

    # KPIs (show as metrics in rows)
    st.markdown("### Key performance indicators")
    k_items = list(summary.items())
    cols_metrics = st.columns(4)
    for i, (k, v) in enumerate(k_items):
        cols_metrics[i % 4].metric(k, f"{v:.2f}" if isinstance(v, float) else str(v))

    # Table + Gantt
    tcol, gcol = st.columns([1, 1])
    with tcol:
        st.markdown("**Per-job outcomes (based on completion on last machine)**")
        st.dataframe(per_job, use_container_width=True, hide_index=True)

        st.download_button(
            "Download per-job results (CSV)",
            data=per_job.to_csv(index=False).encode("utf-8"),
            file_name="schedule_results.csv",
            mime="text/csv",
        )

    with gcol:
        fig = plot_gantt(times, st.session_state.sequence, m)
        st.pyplot(fig, clear_figure=True)