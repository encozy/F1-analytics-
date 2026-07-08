# -*- coding: utf-8 -*-
"""
F1 RACE LAB — анализ любой сессии Формулы-1 (2018+) на реальных данных FastF1.
Запуск:  streamlit run app.py
"""
import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import fastf1
from fastf1 import plotting as f1p
from fastf1 import utils as f1u

# ---------------- базовая настройка ----------------
st.set_page_config(page_title="F1 Race Lab", page_icon="🏎️", layout="wide")

UI = dict(bg="#0C0E12", panel="#141821", line="#232937",
          text="#E8EAF0", mut="#8A93A6", purple="#B57BFF", green="#3DDC97")
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
COMPOUND_FALLBACK = {"SOFT": "#E8112D", "MEDIUM": "#FFD12E", "HARD": "#EDEDE6",
                     "INTERMEDIATE": "#43B02A", "WET": "#0067AD", "UNKNOWN": "#8A93A6"}
SES_RU = {"R": "Гонка", "Q": "Квалификация", "S": "Спринт", "SQ": "Спринт-квалификация",
          "FP1": "Практика 1", "FP2": "Практика 2", "FP3": "Практика 3"}

os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.2rem; }}
  [data-testid="stMetricValue"] {{ font-family: {MONO}; font-size: 1.15rem; }}
  [data-testid="stMetricLabel"] {{ letter-spacing: .12em; text-transform: uppercase; font-size: .7rem; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
</style>
""", unsafe_allow_html=True)

# ---------------- утилиты ----------------
def sec(td):
    """timedelta → секунды (float) c защитой от NaT."""
    try:
        return td.total_seconds()
    except Exception:
        return np.nan

def fmt_lap(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "—"
    m = int(s // 60)
    return f"{m}:{s - 60 * m:06.3f}"

def team_color(team, session):
    try:
        return f1p.get_team_color(str(team), session=session)
    except Exception:
        import hashlib
        h = int(hashlib.md5(str(team).encode()).hexdigest()[:6], 16)
        return f"#{h:06x}"

def compound_color(c, session):
    try:
        return f1p.get_compound_color(str(c), session=session)
    except Exception:
        return COMPOUND_FALLBACK.get(str(c).upper(), "#8A93A6")

def pick_drv(laps, abbr):
    try:
        return laps.pick_drivers(abbr)
    except AttributeError:
        return laps.pick_driver(abbr)

def dark(fig, h=380, title=None):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=UI["panel"], plot_bgcolor=UI["panel"],
        height=h, margin=dict(l=10, r=10, t=40 if title else 16, b=10),
        title=title, font=dict(family=MONO, size=12),
        legend=dict(orientation="h", y=1.06, x=0),
    )
    fig.update_xaxes(gridcolor=UI["line"], zerolinecolor=UI["line"])
    fig.update_yaxes(gridcolor=UI["line"], zerolinecolor=UI["line"])
    return fig

@st.cache_data(show_spinner=False, ttl=86400)
def get_events(year: int):
    sch = fastf1.get_event_schedule(year, include_testing=False)
    return list(sch["EventName"])

@st.cache_resource(show_spinner=False)
def load_session(year: int, gp: str, ses: str, telemetry: bool):
    s = fastf1.get_session(year, gp, ses)
    s.load(laps=True, telemetry=telemetry, weather=False, messages=False)
    return s

# ---------------- сайдбар ----------------
with st.sidebar:
    st.markdown("### 🏁 F1 RACE LAB")
    st.caption("Реальные данные FastF1 · любая сессия с 2018 года")
    year = st.selectbox("Сезон", list(range(2026, 2017, -1)), index=1)
    try:
        events = get_events(year)
    except Exception:
        st.error("Не удалось получить календарь. Проверь интернет и попробуй ещё раз.")
        st.stop()
    gp = st.selectbox("Гран-при", events)
    ses = st.selectbox("Сессия", list(SES_RU.keys()), format_func=lambda x: SES_RU[x])
    tel_on = st.toggle("Телеметрия", value=True,
                       help="Скорость по дистанции, дельта, аэро-скэттер. Первая загрузка сессии дольше.")
    if st.button("Загрузить сессию", type="primary", use_container_width=True):
        st.session_state["key"] = (year, gp, ses, tel_on)
    st.caption("Первая загрузка сессии — 1–3 мин (качаются официальные тайминги), дальше мгновенно из кэша.")

if "key" not in st.session_state:
    st.markdown("## F1 Race Lab")
    st.info("⬅️ Выбери сезон, Гран-при и сессию, затем нажми «Загрузить сессию».")
    st.stop()

key = st.session_state["key"]
try:
    with st.spinner(f"Гружу {key[1]} {key[0]} · {SES_RU[key[2]]} — первый раз это 1–3 минуты…"):
        session = load_session(*key)
except Exception as e:
    st.error(f"Сессию загрузить не удалось: {e}")
    st.caption("Частые причины: сессия ещё не прошла, для этого уик-энда нет такого типа сессии (например, спринта), или нет интернета.")
    st.stop()

laps = session.laps.copy()
if laps is None or len(laps) == 0:
    st.warning("В этой сессии нет кругов.")
    st.stop()

# подготовка колонок
laps["LapSec"] = laps["LapTime"].apply(sec)
for i in (1, 2, 3):
    laps[f"S{i}"] = laps[f"Sector{i}Time"].apply(sec)
laps["Clean"] = (laps["PitInTime"].isna() & laps["PitOutTime"].isna()
                 & laps["LapSec"].notna() & (laps["TrackStatus"].astype(str) == "1"))

results = session.results if session.results is not None else pd.DataFrame()
have_res = len(results) > 0 and "Abbreviation" in results.columns

if have_res:
    order = [a for a in results["Abbreviation"] if a in set(laps["Driver"])]
    team_of = dict(zip(results["Abbreviation"], results["TeamName"]))
else:
    fl_by = laps.groupby("Driver")["LapSec"].min().sort_values()
    order = list(fl_by.index)
    team_of = laps.groupby("Driver")["Team"].first().to_dict()

col_of = {a: team_color(team_of.get(a, ""), session) for a in order}
mate_ix = {}
_seen = {}
for a in order:
    t = team_of.get(a, "")
    mate_ix[a] = _seen.get(t, 0)
    _seen[t] = _seen.get(t, 0) + 1

is_race = key[2] in ("R", "S")
is_quali = key[2] in ("Q", "SQ")

# ---------------- шапка ----------------
ev = session.event
st.markdown(
    f"<div style='letter-spacing:.25em;color:{UI['purple']};font-weight:800;font-size:.75rem'>F1 RACE LAB</div>"
    f"<h2 style='margin:.1rem 0 .2rem'>{ev['EventName']} {year} · {SES_RU[key[2]]}</h2>"
    f"<div style='color:{UI['mut']};font-family:{MONO};font-size:.85rem'>{ev['Location']}, {ev['Country']}</div>",
    unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
fl_row = laps.pick_fastest()
if fl_row is not None and pd.notna(fl_row["LapTime"]):
    m1.metric("Быстрый круг", f"{fl_row['Driver']} · {fmt_lap(sec(fl_row['LapTime']))}")
if have_res and is_race:
    top = results.iloc[0]
    m2.metric("Победитель" if key[2] == "R" else "P1", f"{top['Abbreviation']} · {top['TeamName']}")
elif have_res:
    top = results.iloc[0]
    m2.metric("Поул" if is_quali else "P1", f"{top['Abbreviation']}")
m3.metric("Кругов в данных", f"{int(laps['LapNumber'].max())}")
m4.metric("Пилотов", f"{laps['Driver'].nunique()}")

sel = st.multiselect("Пилоты", order, default=order[:6],
                     help="Выбор влияет на «Темп» и «Шины». Быстрый способ: очисти и добавь нужных.")

tabs = ["🏁 Темп"]
if is_race:
    tabs.append("🛞 Шины и стинты")
tabs.append("📡 Телеметрия")
if is_quali:
    tabs.append("⏱️ Квалификация")
if have_res:
    tabs.append("📋 Итоги")
T = dict(zip(tabs, st.tabs(tabs)))

# ================= ТЕМП =================
with T["🏁 Темп"]:
    c1, c2 = st.columns([3, 2])
    clean_only = c1.checkbox("Только чистые круги (без пит-стопов, SC/VSC)", value=True)

    src = laps[laps["Driver"].isin(sel)]
    src = src[src["Clean"]] if clean_only else src[src["LapSec"].notna()]

    if len(src) == 0:
        st.info("Нет кругов под выбранные фильтры.")
    else:
        med = src.groupby("Driver")["LapSec"].median().sort_values()
        fig = go.Figure()
        for a in med.index:
            vals = src.loc[src["Driver"] == a, "LapSec"]
            fig.add_trace(go.Box(y=vals, name=a, marker_color=col_of[a],
                                 line=dict(width=1.6), boxpoints="outliers",
                                 hovertemplate="%{y:.3f}с<extra>" + a + "</extra>"))
        fig.update_yaxes(title="время круга, с")
        st.plotly_chart(dark(fig, 420, "Гоночный темп — распределение времени круга (сортировка по медиане)"),
                        use_container_width=True)

        if is_race and len(med) > 0:
            ref = float(med.min())
            fig2 = go.Figure()
            for a in sel:
                d = laps[laps["Driver"] == a].sort_values("LapNumber")
                ls = d["LapSec"].fillna(d[["S1", "S2", "S3"]].sum(axis=1, min_count=3))
                ok = ls.notna()
                gap = ref * d.loc[ok, "LapNumber"] - ls[ok].cumsum()
                fig2.add_trace(go.Scatter(
                    x=d.loc[ok, "LapNumber"], y=gap, name=a, mode="lines",
                    line=dict(color=col_of[a], width=2, dash="dash" if mate_ix.get(a) else "solid"),
                    hovertemplate="круг %{x} · %{y:+.1f}с<extra>" + a + "</extra>"))
            fig2.update_xaxes(title="круг")
            fig2.update_yaxes(title="против эталонного темпа, с")
            st.plotly_chart(dark(fig2, 420, "Ход гонки (race trace): выше — быстрее эталона, ступени вниз — пит-стопы"),
                            use_container_width=True)

# ================= ШИНЫ =================
if "🛞 Шины и стинты" in T:
    with T["🛞 Шины и стинты"]:
        stints = (laps.groupby(["Driver", "Stint", "Compound"])
                  .agg(frm=("LapNumber", "min"), to=("LapNumber", "max"))
                  .reset_index())
        fig = go.Figure()
        seen_comp = set()
        for a in order:
            for _, s in stints[stints["Driver"] == a].iterrows():
                comp = str(s["Compound"]).upper()
                fig.add_trace(go.Bar(
                    y=[a], x=[s["to"] - s["frm"] + 1], base=[s["frm"] - 1],
                    orientation="h", marker_color=compound_color(comp, session),
                    marker_line=dict(color=UI["bg"], width=1),
                    name=comp, showlegend=comp not in seen_comp,
                    text=f"{comp[:1]}·{int(s['to'] - s['frm'] + 1)}",
                    textposition="inside", insidetextanchor="middle",
                    textfont=dict(color="#14161C", family=MONO, size=10),
                    hovertemplate=f"{a} · {comp}: круги %{{base}}–%{{x}}<extra></extra>"))
                seen_comp.add(comp)
        fig.update_layout(barmode="stack", yaxis=dict(categoryorder="array",
                                                      categoryarray=list(reversed(order)), title=""))
        fig.update_xaxes(title="круг")
        st.plotly_chart(dark(fig, max(420, 24 * len(order)), "Лента стинтов (порядок — итоговый протокол)"),
                        use_container_width=True)

        st.divider()
        fk = st.slider("Поправка на топливо, с/круг", 0.0, 0.10, 0.055, 0.005,
                       help="Вычитается эффект лёгкой машины, чтобы виден был чистый износ резины.")
        total = laps["LapNumber"].max()
        d2 = laps[(laps["Driver"].isin(sel)) & laps["Clean"]].copy()
        d2["Corr"] = d2["LapSec"] - fk * (total - d2["LapNumber"])
        fig3 = go.Figure()
        info = []
        for comp in d2["Compound"].dropna().unique():
            sub = d2[d2["Compound"] == comp]
            cc = compound_color(comp, session)
            fig3.add_trace(go.Scatter(x=sub["TyreLife"], y=sub["Corr"], mode="markers",
                                      name=str(comp), marker=dict(color=cc, size=6, opacity=0.65),
                                      customdata=sub["Driver"],
                                      hovertemplate="%{customdata} · %{x} кр. · %{y:.3f}с<extra></extra>"))
            ok = sub[["TyreLife", "Corr"]].dropna()
            if len(ok) >= 6:
                k, b = np.polyfit(ok["TyreLife"], ok["Corr"], 1)
                xs = np.array([ok["TyreLife"].min(), ok["TyreLife"].max()])
                fig3.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines",
                                          line=dict(color=cc, width=2.5), showlegend=False,
                                          hoverinfo="skip"))
                info.append(f"<span style='color:{cc}'>{comp}: {k:+.3f} с/круг</span>")
        fig3.update_xaxes(title="возраст комплекта, кругов")
        fig3.update_yaxes(title="время круга без топлива, с")
        st.plotly_chart(dark(fig3, 420, "Деградация резины (наклон линии = темп деградации)"),
                        use_container_width=True)
        if info:
            st.markdown(" · ".join(info), unsafe_allow_html=True)

# ================= ТЕЛЕМЕТРИЯ =================
if "📡 Телеметрия" in T:
    with T["📡 Телеметрия"]:
        # Speed trap — из данных кругов, телеметрия не требуется
        trap = (laps.groupby("Driver")["SpeedST"].max().dropna()
                .reindex(order).dropna().sort_values(ascending=True))
        if len(trap):
            figt = go.Figure(go.Bar(
                x=trap.values, y=trap.index, orientation="h",
                marker_color=[col_of.get(a, "#888") for a in trap.index],
                text=[f"{v:.0f}" for v in trap.values], textposition="outside",
                textfont=dict(family=MONO, size=10)))
            figt.update_xaxes(title="км/ч (speed trap)")
            st.plotly_chart(dark(figt, max(360, 22 * len(trap)), "Speed trap — максимум за сессию"),
                            use_container_width=True)
        st.divider()

        if not key[3]:
            st.info("Скорость по дистанции, дельта и аэро-скэттер требуют телеметрии: "
                    "включи тумблер «Телеметрия» слева и загрузи сессию заново. "
                    "На бесплатном облаке гонки с телеметрией могут не влезать в память — "
                    "квалификации и спринты обычно проходят.")
        else:
            try:
                ca, cb = st.columns(2)
                A = ca.selectbox("Пилот A", order, index=0)
                B = cb.selectbox("Пилот B", order, index=min(1, len(order) - 1))
                lapA = pick_drv(laps, A).pick_fastest()
                lapB = pick_drv(laps, B).pick_fastest()
                if lapA is None or lapB is None:
                    st.info("У одного из пилотов нет полного быстрого круга.")
                else:
                    tA = lapA.get_car_data().add_distance()
                    tB = lapB.get_car_data().add_distance()
                    st.markdown(
                        f"<span style='font-family:{MONO}'>"
                        f"<b style='color:{col_of[A]}'>{A}</b> {fmt_lap(sec(lapA['LapTime']))} · "
                        f"<b style='color:{col_of[B]}'>{B}</b> {fmt_lap(sec(lapB['LapTime']))} · "
                        f"Δ {sec(lapB['LapTime']) - sec(lapA['LapTime']):+.3f}с</span>",
                        unsafe_allow_html=True)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=tA["Distance"], y=tA["Speed"], name=A,
                                             line=dict(color=col_of[A], width=2)))
                    fig.add_trace(go.Scatter(x=tB["Distance"], y=tB["Speed"], name=B,
                                             line=dict(color=col_of[B], width=2,
                                                       dash="dash" if team_of.get(A) == team_of.get(B) else "solid")))
                    try:
                        corners = session.get_circuit_info().corners
                        for _, c in corners.iterrows():
                            fig.add_vline(x=c["Distance"], line_color=UI["line"], line_width=1)
                            fig.add_annotation(x=c["Distance"], y=1.02, yref="paper", showarrow=False,
                                               text=f"{int(c['Number'])}{c.get('Letter', '') or ''}",
                                               font=dict(size=9, color=UI["mut"]))
                    except Exception:
                        pass
                    fig.update_xaxes(title="дистанция, м")
                    fig.update_yaxes(title="км/ч")
                    st.plotly_chart(dark(fig, 380, "Скорость по дистанции — быстрые круги"),
                                    use_container_width=True)

                    try:
                        delta, ref_tel, _ = f1u.delta_time(lapA, lapB)
                        figd = go.Figure()
                        figd.add_trace(go.Scatter(x=ref_tel["Distance"], y=delta,
                                                  line=dict(color=UI["purple"], width=2), name="Δ"))
                        figd.add_hline(y=0, line_color=UI["mut"])
                        figd.update_xaxes(title="дистанция, м")
                        figd.update_yaxes(title=f"Δ, с (выше нуля — {B} теряет)")
                        st.plotly_chart(dark(figd, 260, f"Дельта времени: {B} относительно {A}"),
                                        use_container_width=True)
                    except Exception:
                        st.caption("Дельту для этой пары посчитать не удалось (неполные данные круга).")

                    with st.spinner("Считаю аэро-скэттер по командам…"):
                        rows = []
                        try:
                            cdist = session.get_circuit_info().corners["Distance"].values
                        except Exception:
                            cdist = None
                        for tm in pd.unique([team_of[a] for a in order]):
                            tl = laps[laps["Team"] == tm]
                            fl = tl.pick_fastest() if len(tl) else None
                            if fl is None or pd.isna(fl["LapTime"]):
                                continue
                            try:
                                cd = fl.get_car_data().add_distance()
                            except Exception:
                                continue
                            vmax = float(cd["Speed"].max())
                            if cdist is not None and len(cdist):
                                mins = []
                                for x0 in cdist:
                                    w = cd[(cd["Distance"] > x0 - 70) & (cd["Distance"] < x0 + 70)]
                                    if len(w):
                                        mins.append(float(w["Speed"].min()))
                                slow = [m for m in mins if m < 230]
                                vcor = float(np.mean(slow)) if slow else float(np.percentile(cd["Speed"], 8))
                            else:
                                vcor = float(np.percentile(cd["Speed"], 8))
                            rows.append((tm, vcor, vmax))
                    if rows:
                        figa = go.Figure()
                        for tm, x, y in rows:
                            cc = team_color(tm, session)
                            figa.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text",
                                                      marker=dict(color=cc, size=11),
                                                      text=[str(tm)], textposition="middle right",
                                                      textfont=dict(size=10, color=UI["mut"]),
                                                      showlegend=False,
                                                      hovertemplate=f"{tm}<br>повороты %{{x:.0f}} · пик %{{y:.0f}}<extra></extra>"))
                        figa.update_xaxes(title="ср. скорость в медленных поворотах, км/ч")
                        figa.update_yaxes(title="V-max, км/ч")
                        st.plotly_chart(dark(figa, 380, "Прижимная сила (косвенно): выше-левее — «низкое крыло»"),
                                        use_container_width=True)
            except Exception as e:
                st.warning(f"Телеметрия недоступна: {e}. Перезагрузи сессию с включённым тумблером «Телеметрия».")

# ================= КВАЛИФИКАЦИЯ =================
if "⏱️ Квалификация" in T:
    with T["⏱️ Квалификация"]:
        if have_res and {"Q1", "Q2", "Q3"}.issubset(results.columns):
            q = results.copy()
            for c in ("Q1", "Q2", "Q3"):
                q[c] = q[c].apply(sec)
            q["Best"] = q[["Q1", "Q2", "Q3"]].min(axis=1)
            q = q[q["Best"].notna()].sort_values("Best")
            pole = q["Best"].iloc[0]
            figq = go.Figure(go.Bar(
                x=(q["Best"] - pole).values, y=q["Abbreviation"], orientation="h",
                marker_color=[col_of.get(a, "#888") for a in q["Abbreviation"]],
                text=[fmt_lap(v) for v in q["Best"]], textposition="outside",
                textfont=dict(family=MONO, size=10),
                hovertemplate="+%{x:.3f}с<extra></extra>"))
            figq.update_yaxes(autorange="reversed")
            figq.update_xaxes(title="отставание от поула, с")
            st.plotly_chart(dark(figq, max(420, 22 * len(q)), "Итог квалификации (лучшее время из Q1–Q3)"),
                            use_container_width=True)

        acc = laps[laps["LapSec"].notna()]
        best = acc.groupby("Driver")[["S1", "S2", "S3"]].min()
        best = best.reindex([a for a in order if a in best.index]).dropna(how="all")
        best["Теор. круг"] = best[["S1", "S2", "S3"]].sum(axis=1)
        best = best.sort_values("Теор. круг")
        sess_best = best[["S1", "S2", "S3"]].min()

        def hl(col):
            return ["background-color:#B57BFF33;color:#fff" if abs(v - sess_best[col.name]) < 1e-9 else ""
                    for v in col]

        st.markdown("**Лучшие сектора** — фиолетовым выделен лучший сектор сессии; "
                    "теоретический круг = сумма личных лучших секторов.")
        st.dataframe(
            best.style.apply(hl, subset=["S1", "S2", "S3"]).format("{:.3f}"),
            use_container_width=True, height=min(600, 40 + 35 * len(best)))

# ================= ИТОГИ =================
if "📋 Итоги" in T:
    with T["📋 Итоги"]:
        cols = [c for c in ["Position", "Abbreviation", "FullName", "TeamName",
                            "GridPosition", "Time", "Status", "Points"] if c in results.columns]
        r = results[cols].copy()
        ren = {"Position": "P", "Abbreviation": "Код", "FullName": "Пилот", "TeamName": "Команда",
               "GridPosition": "Старт", "Time": "Время/гэп", "Status": "Статус", "Points": "Очки"}
        r = r.rename(columns=ren)
        if "Время/гэп" in r.columns:
            r["Время/гэп"] = r["Время/гэп"].apply(lambda v: str(v).replace("0 days ", "") if pd.notna(v) else "")
        st.dataframe(r, use_container_width=True, hide_index=True,
                     height=min(760, 40 + 35 * len(r)))

st.caption("Данные: официальный тайминг F1 через FastF1. Кэш — в папке ./cache (можно удалять). "
           "F1 Race Lab — учебно-аналитический проект, не аффилирован с Formula 1.")
