"""UK Energy Transition — integrated analytical brief (Sections II–XI).

Whole-system view: LCREE (ONS), MCS batteries, RHI, DESNZ generation /
consumption / load factors / solar costs / price volatility, plus IRENA
benchmarks.  Derived facts live in ``cleaned_data/derived/`` — run
``python3 scripts/build_derived_facts.py`` after cleaning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib import load_sheet  # noqa: E402
from analysis.cpi import deflate_to_base, load_cpi  # noqa: E402
from analysis.integrated_productivity import (  # noqa: E402
    RENEWABLE_SECTOR_KEYWORDS,
    renewable_sector_aggregate,
    rolling_3y,
)
from analysis.irena_uk_benchmarks import (  # noqa: E402
    DEFAULT_PEER_ISO3,
    UK_ISO3,
    cagr,
    load_optional_reference,
    merge_sdg_oecd_series,
    sdg_series_country,
    top_n_countries_by_cf,
    weighted_capacity_factor,
)
from analysis.learning_rate import fit_curve_for_plot, fit_learning_rate  # noqa: E402
from analysis.palettes import plotly_layout_defaults  # noqa: E402
from analysis.rhi_metrics import heat_output_per_install, queue_pipeline, scheme_admin_actuals  # noqa: E402
from analysis.stats_breaks import chow_mean_shift, ruptures_peaks  # noqa: E402
from analysis.time_utils import hours_per_calendar_year  # noqa: E402
from analysis.volatility import annual_volatility_pivot, coerce_monthly, rolling_volatility  # noqa: E402


def _scalar_finite(x) -> float | None:
    if x is None:
        return None
    v = pd.to_numeric(x, errors="coerce")
    if isinstance(v, pd.Series):
        v = v.iloc[0] if len(v.index) else np.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


st.set_page_config(
    page_title="UK Energy Transition · Integrated Brief",
    layout="wide",
)

st.title("UK Energy Transition — integrated analytical brief")
st.caption(
    "Economic activity (LCREE) · physical deployment (MCS, RHI, generation) · "
    "technical performance (FIT load factors) · market context (costs, volatility, schemes)."
)

# -------- Data loads ---------------------------------------------------------
country = load_sheet("IRENA", "IRENA_Statistics_Extract_2025H2__Country")
cf_tbl = load_sheet("derived", "capacity_factor")
uk_fact = load_sheet("derived", "uk_renewables_fact")
lcree = load_sheet("derived", "lcree_productivity")
annual_lf = load_sheet(
    "DESNZ",
    "Annual_and_quarterly_load_factors_FIT_years_2-15__Annual_load_factors",
)
elec_annual = load_sheet("derived", "electricity_generation_annual")
renew_ind = load_sheet("derived", "renewable_share_by_industry")
rhi_unit = load_sheet("derived", "rhi_unit_economics")
solar_lr = load_sheet("derived", "solar_learning")
price_vol = load_sheet("derived", "price_volatility_annual")
mcs_m = load_sheet("derived", "mcs_battery_metrics")
lcree_country = load_sheet("derived", "lcree_by_country")
integrated = load_sheet("derived", "integrated_productivity")
pop_ref = load_optional_reference("population.csv")

solar_newretro = load_sheet("DESNZ", "Solar_Costs_2024-25__New_build_and_retrofit_costs")
price_m = load_sheet("DESNZ", "price-volatility-of-gas__Ark1")
non_dom = load_sheet("DESNZ", "non-domestic-renewable-h__Ark1")
scheme_adm = load_sheet("Ofgem", "ofgem-scheme-administrat__Ark1")
heat_tbl = load_sheet("DESNZ", "10energyconsumptionheat__Heat")
rhi_approved = load_sheet("Ofgem", "total-number-of-approved__Ark1")
renew09 = load_sheet("DESNZ", "09energyconsumptionrenewableandwastesources__Renewables")
fte_ind = load_sheet("ONS", "lcreedataset2024__LCREE_FTE_by_industry")
lc250 = load_sheet("ONS", "lcreedataset2024__LCREE_250_businesses")

cpi_df = load_cpi()
if cpi_df.empty:
    st.warning(
        "Nominal £ for solar costs and RHI payments — add `config/reference/cpi_uk.csv` "
        "(see `cpi_uk.csv.example`) to deflate to 2024 prices for learning-rate comparisons."
    )

if country.empty:
    st.error("IRENA Country sheet missing — run the cleaning pipeline.")
    st.stop()

for c in (
    "year",
    "electricity_installed_capacity_mw",
    "electricity_generation_gwh",
    "sdg_7b1_re_capacity_per_capita_w_inhabitant",
    "public_flows_2022_usd_m",
    "heat_generation_tj",
):
    if c in country.columns:
        country[c] = pd.to_numeric(country[c], errors="coerce")
country["iso3_code"] = country["iso3_code"].astype(str)
iso_to_name = (
    country.dropna(subset=["iso3_code", "country"])
    .drop_duplicates("iso3_code")
    .set_index("iso3_code")["country"]
    .to_dict()
)

years_all = sorted(country["year"].dropna().unique().astype(int).tolist())
y_lo, y_hi = int(years_all[0]), int(years_all[-1])

with st.sidebar:
    st.subheader("Scope")
    yr = st.slider("Year range", min_value=y_lo, max_value=y_hi, value=(max(y_lo, 2000), y_hi))
    peer_opts = [iso_to_name.get(i, i) for i in DEFAULT_PEER_ISO3 if i in iso_to_name]
    peer_labels_sel = st.multiselect(
        "Peer countries (IRENA tab)",
        sorted({iso_to_name.get(k, k) for k in iso_to_name.keys()}),
        default=[iso_to_name[i] for i in DEFAULT_PEER_ISO3 if i in iso_to_name],
    )
    peer_iso3 = [k for k, v in iso_to_name.items() if v in peer_labels_sel]
    tech_opts = sorted(country["technology"].dropna().unique().tolist())
    default_tech = [
        t
        for t in (
            "Solar photovoltaic",
            "Onshore wind energy",
            "Offshore wind energy",
            "Renewable hydropower",
            "Biogas",
        )
        if t in tech_opts
    ]
    tech_sel = st.multiselect("Technologies (IRENA tab)", tech_opts, default=default_tech or tech_opts[:5])
    sector_kw = st.text_input(
        "LCREE sector keywords (productivity tab)",
        value="|".join(RENEWABLE_SECTOR_KEYWORDS),
        help="Pipe-separated substrings matched against LCREE sector names.",
    )
    kw_tuple = tuple(k.strip() for k in sector_kw.split("|") if k.strip())

uk_long = country.loc[country["iso3_code"] == UK_ISO3].copy()
uk_long = uk_long[uk_long["year"].between(yr[0], yr[1])]
if tech_sel:
    uk_long = uk_long[uk_long["technology"].isin(tech_sel)]

uk_fact_y = uk_fact.copy()
if not uk_fact_y.empty:
    uk_fact_y["year"] = pd.to_numeric(uk_fact_y["year"], errors="coerce")
    uk_fact_y = uk_fact_y[uk_fact_y["year"].between(yr[0], yr[1])]
if tech_sel:
    uk_fact_y = uk_fact_y[uk_fact_y["technology"].isin(tech_sel)]

uk_fact_agg = (
    uk_fact_y.groupby(["year", "technology"], as_index=False)
    .agg(
        electricity_generation_gwh=("electricity_generation_gwh", "sum"),
        electricity_installed_capacity_mw=("electricity_installed_capacity_mw", "sum"),
    )
)
uk_fact_agg["hours"] = uk_fact_agg["year"].map(
    lambda y: hours_per_calendar_year(int(y)) if pd.notna(y) else np.nan
)
uk_fact_agg["cf"] = uk_fact_agg["electricity_generation_gwh"] / (
    uk_fact_agg["electricity_installed_capacity_mw"] * uk_fact_agg["hours"] / 1000.0
)

uk_cap_total = (
    uk_fact_y.dropna(subset=["electricity_installed_capacity_mw"])
    .groupby("year")["electricity_installed_capacity_mw"]
    .sum()
)

pre_lo, pre_hi = yr[0], min(2014, yr[1])
post_lo, post_hi = max(2015, yr[0]), yr[1]
cap_pre_g = (
    uk_fact_y.loc[uk_fact_y["year"].between(pre_lo, pre_hi)]
    .groupby("year")["electricity_installed_capacity_mw"]
    .sum()
    .sort_index()
)
cap_post_g = (
    uk_fact_y.loc[uk_fact_y["year"].between(post_lo, post_hi)]
    .groupby("year")["electricity_installed_capacity_mw"]
    .sum()
    .sort_index()
)

sdg_uk = sdg_series_country(UK_ISO3, country)
sdg_oecd = merge_sdg_oecd_series(country, pop_ref if not pop_ref.empty else None)

if "alignment_figs" not in st.session_state:
    st.session_state["alignment_figs"] = {}
sf = st.session_state["alignment_figs"]
sf.clear()


def register(name: str, fig) -> None:
    sf[name] = fig


# -------- Tab layout ---------------------------------------------------------
tabs = st.tabs(
    [
        "Executive summary",
        "Physical performance",
        "Costs & learning",
        "Market & risk",
        "Regional & structural",
        "Integrated productivity",
        "Structural breaks",
        "Risk flags",
        "IRENA benchmarks",
        "Annex & methodology",
    ]
)

# ----- Tab 0 ----------------------------------------------------------------
with tabs[0]:
    st.markdown("### Executive summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    if not elec_annual.empty:
        ea = elec_annual.copy()
        ea["year"] = pd.to_numeric(ea["year"], errors="coerce")
        last = ea.sort_values("year").iloc[-1]
        c1.metric(
            "Renewable share of electricity (%)",
            f"{_scalar_finite(last.get('renewable_share_pct')):.1f}"
            if _scalar_finite(last.get("renewable_share_pct")) is not None
            else "n/a",
        )
    else:
        c1.metric("Renewable share of electricity (%)", "n/a")

    cf_ready = uk_fact_y.dropna(
        subset=["electricity_generation_gwh", "electricity_installed_capacity_mw"]
    )
    cf_ready = cf_ready.loc[cf_ready["electricity_generation_gwh"].astype(float) > 0]
    last_cf_y = (
        int(pd.to_numeric(cf_ready["year"], errors="coerce").max())
        if not cf_ready.empty
        else None
    )
    if last_cf_y is not None:
        last_rows = cf_ready.loc[pd.to_numeric(cf_ready["year"], errors="coerce") == float(last_cf_y)]
        wcf = weighted_capacity_factor(
            last_rows["electricity_generation_gwh"],
            last_rows["electricity_installed_capacity_mw"],
            last_cf_y,
        )
        c2.metric(
            f"UK weighted CF ({last_cf_y})",
            f"{wcf:.3f}" if np.isfinite(wcf) else "n/a",
        )
    else:
        c2.metric("UK weighted CF", "n/a")

    if not integrated.empty:
        ip = integrated.copy()
        ip["year"] = pd.to_numeric(ip["year"], errors="coerce")
        ip = ip.sort_values("year")
        # Skip the last year if the renewable-FTE coverage looks much smaller
        # than the rolling median (LCREE 2024 release lacks the Electricity
        # supply sector).
        med = float(pd.to_numeric(ip["renewable_fte"], errors="coerce").median())
        ip_ok = ip[pd.to_numeric(ip["renewable_fte"], errors="coerce") >= 0.4 * med]
        ly = ip_ok.iloc[-1] if not ip_ok.empty else ip.iloc[-1]
        kpi_year = int(ly["year"]) if pd.notna(ly["year"]) else "?"
        c3.metric(
            f"Turnover / GWh ({kpi_year}, £k)",
            f"{_scalar_finite(ly.get('turnover_per_gwh_thousand_gbp')):.1f}"
            if _scalar_finite(ly.get("turnover_per_gwh_thousand_gbp")) is not None
            else "n/a",
        )
        c4.metric(
            f"FTE / TWh ({kpi_year})",
            f"{_scalar_finite(ly.get('fte_per_twh')):.0f}"
            if _scalar_finite(ly.get("fte_per_twh")) is not None
            else "n/a",
        )
    else:
        c3.metric("Turnover / GWh", "n/a")
        c4.metric("FTE / TWh", "n/a")

    nd = queue_pipeline(non_dom)
    if not nd.empty and "queue" in nd.columns:
        c5.metric("Non-domestic RHI queue (latest)", f"{int(nd['queue'].iloc[-1]):,}")
    else:
        c5.metric("Non-domestic RHI queue", "n/a")

    st.markdown(
        "**Analytical framing** — test whether economic value creation in LCREE "
        "tracks physical delivery (generation, installs), technical performance "
        "(load factors), and market signals (costs, volatility, scheme uptake)."
    )

# ----- Tab 1 ----------------------------------------------------------------
with tabs[1]:
    st.subheader("Physical performance — generation, industry renewables, load factors")
    ea = pd.DataFrame()
    if not elec_annual.empty:
        ea = elec_annual.copy()
        ea["year"] = pd.to_numeric(ea["year"], errors="coerce")
        ea = ea[ea["year"].between(yr[0], yr[1])]
        fig1 = px.area(
            ea,
            x="year",
            y="renewable_share_pct",
            labels={"renewable_share_pct": "Renewable share of electricity (%)"},
        )
        fig1.update_layout(**plotly_layout_defaults(), height=380)
        st.plotly_chart(fig1, width="stretch", key="brief_renewable_share_electricity")
        register("renewable_share_electricity", fig1)

    if not lcree.empty and not ea.empty:
        re_agg = renewable_sector_aggregate(lcree, kw_tuple)
        re_agg = re_agg[re_agg["year"].between(yr[0], yr[1])]
        m = ea.merge(re_agg, on="year", how="inner")
        if not m.empty:
            fig2 = go.Figure()
            fig2.add_trace(
                go.Bar(x=m["year"], y=m["renewable_share_pct"], name="Renewable share (%)", yaxis="y")
            )
            fig2.add_trace(
                go.Scatter(
                    x=m["year"],
                    y=m["renewable_fte"],
                    name="LCREE FTE (renewable-relevant sectors)",
                    yaxis="y2",
                    mode="lines+markers",
                )
            )
            fig2.update_layout(
                **plotly_layout_defaults(),
                height=420,
                yaxis=dict(title="Renewable share (%)"),
                yaxis2=dict(title="FTE", overlaying="y", side="right", showgrid=False),
            )
            st.plotly_chart(fig2, width="stretch", key="brief_renewable_share_fte_overlay")
            register("renewable_share_fte_overlay", fig2)

    if not annual_lf.empty and "weighted_mean" in annual_lf.columns:
        lf = annual_lf.copy()
        lf["fy_start"] = pd.to_numeric(lf["financial_year"].astype(str).str[:4], errors="coerce")
        lf = lf[lf["fy_start"].between(yr[0], yr[1])]
        fig3 = px.line(
            lf.sort_values(["technology", "fy_start"]),
            x="fy_start",
            y="weighted_mean",
            color="technology",
            labels={"weighted_mean": "Weighted mean load factor (%)", "fy_start": "FY start year"},
        )
        fig3.update_layout(**plotly_layout_defaults(), height=420)
        if "coverage" in lf.columns:
            st.caption("FIT load factors include a **coverage** field — treat low-coverage years cautiously.")
        st.plotly_chart(fig3, width="stretch", key="brief_annual_load_factors")
        register("annual_load_factors", fig3)

    if not renew_ind.empty:
        ri = renew_ind.copy()
        ri["year"] = pd.to_numeric(ri["year"], errors="coerce")
        ri = ri[ri["year"].between(yr[0], yr[1])]
        top = (
            ri.groupby("industry", as_index=False)["renewable_share_pct"]
            .mean()
            .nlargest(12, "renewable_share_pct")
        )
        sub = ri[ri["industry"].isin(top["industry"])]
        fig4 = px.line(
            sub.sort_values(["industry", "year"]),
            x="year",
            y="renewable_share_pct",
            color="industry",
            labels={"renewable_share_pct": "Renewable share of industry energy (%)"},
        )
        fig4.update_layout(**plotly_layout_defaults(), height=460)
        st.plotly_chart(fig4, width="stretch", key="brief_renewable_share_industry")
        register("renewable_share_industry", fig4)

    ho = heat_output_per_install(rhi_approved, heat_tbl)
    if not ho.empty:
        fig5 = px.line(
            ho.sort_values("year"),
            x="year",
            y="heat_mtoe_per_install",
            markers=True,
            labels={"heat_mtoe_per_install": "Heat from renewables (Mtoe) / cumulative approved installs"},
        )
        fig5.update_layout(**plotly_layout_defaults(), height=360)
        st.plotly_chart(fig5, width="stretch", key="brief_heat_per_install")
        register("heat_per_install", fig5)

# ----- Tab 2 ----------------------------------------------------------------
with tabs[2]:
    st.subheader("Technology costs & learning rates")
    if not solar_lr.empty:
        sl = solar_lr.copy()
        sl["calendar_year"] = pd.to_numeric(sl["calendar_year"], errors="coerce")
        sl = sl[sl["calendar_year"].between(yr[0], yr[1])]
        cost_col = "median_cost_gbp_per_kw"
        if not cpi_df.empty:
            sl["cost_real"] = deflate_to_base(sl[cost_col], sl["calendar_year"], cpi_df)
            cost_plot = "cost_real"
            cost_label = "Median £/kW (2024 prices)"
        else:
            sl["cost_real"] = sl[cost_col]
            cost_plot = "cost_real"
            cost_label = "Median £/kW (nominal)"
        for band in sl["band"].dropna().unique():
            sub = sl[sl["band"] == band].dropna(subset=["cum_capacity_mw", cost_plot])
            if len(sub) < 2:
                continue
            fit = fit_learning_rate(sub["cum_capacity_mw"], sub[cost_plot])
            fig = px.scatter(
                sub,
                x="cum_capacity_mw",
                y=cost_plot,
                hover_data=["financial_year"],
                labels={"cum_capacity_mw": "Cumulative UK PV capacity (MW)", cost_plot: cost_label},
                title=f"{band} — LR per doubling: {fit['lr_per_doubling']:.1%}" if np.isfinite(fit["lr_per_doubling"]) else band,
            )
            curve = fit_curve_for_plot(sub["cum_capacity_mw"], sub[cost_plot])
            if not curve.empty:
                fig.add_trace(
                    go.Scatter(x=curve["cum_capacity"], y=curve["cost_fit"], mode="lines", name="log-log fit")
                )
            fig.update_layout(**plotly_layout_defaults(), height=400)
            st.plotly_chart(fig, width="stretch", key=f"brief_solar_learning_{band}")
            register(f"solar_learning_{band}", fig)

    if not solar_newretro.empty:
        st.markdown("**Retrofit vs new build (2024/25 domestic)**")
        sn = solar_newretro.copy()
        if "all_installations_included_in_analysis_mean_kw" in sn.columns:
            figb = px.bar(
                sn,
                x="installation_type",
                y="all_installations_included_in_analysis_mean_kw",
                labels={"all_installations_included_in_analysis_mean_kw": "Mean £/kW"},
            )
            figb.update_layout(**plotly_layout_defaults(), height=360)
            st.plotly_chart(figb, width="stretch", key="brief_solar_retrofit_newbuild")
            register("solar_retrofit_newbuild", figb)

    if not mcs_m.empty:
        mm = mcs_m.copy()
        mm["calendar_year"] = pd.to_numeric(mm.get("calendar_year"), errors="coerce")
        figm = px.line(
            mm.sort_values(["calendar_year", "month_of_installation_note_5"]),
            x="calendar_year",
            y="kwh_per_install",
            markers=True,
            labels={"kwh_per_install": "kWh per install (sample)"},
        )
        figm.update_layout(**plotly_layout_defaults(), height=360)
        st.plotly_chart(figm, width="stretch", key="brief_mcs_kwh_per_install")
        register("mcs_kwh_per_install", figm)

    if not rhi_unit.empty:
        ru = rhi_unit.copy()
        ru["year"] = pd.to_numeric(ru["year"], errors="coerce")
        ru = ru[ru["year"].between(yr[0], yr[1])]
        if not cpi_df.empty:
            ru["cost_plot"] = deflate_to_base(ru["cost_per_install_gbp"], ru["year"], cpi_df)
            rhi_label = "RHI £ per new install (2024 prices)"
        else:
            ru["cost_plot"] = ru["cost_per_install_gbp"]
            rhi_label = "RHI £ per new install (nominal)"
        figr = px.line(
            ru.sort_values(["technology", "year"]),
            x="year",
            y="cost_plot",
            color="technology",
            labels={"cost_plot": rhi_label},
        )
        figr.update_layout(**plotly_layout_defaults(), height=400)
        st.plotly_chart(figr, width="stretch", key="brief_rhi_cost_per_install")
        register("rhi_cost_per_install", figr)

# ----- Tab 3 ----------------------------------------------------------------
with tabs[3]:
    st.subheader("Market dynamics — volatility, acquisitions, RHI pipeline, scheme admin")
    pv_pivot = annual_volatility_pivot(price_vol, "cv")
    if not pv_pivot.empty:
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=pv_pivot.values.astype(float),
                x=pv_pivot.columns.astype(str),
                y=pv_pivot.index.astype(str),
                colorscale="RdYlGn_r",
                zmin=0,
                zmax=float(np.nanpercentile(pv_pivot.to_numpy(dtype=float), 95))
                if np.isfinite(pv_pivot.to_numpy(dtype=float)).any()
                else 1.0,
            )
        )
        fig_hm.update_layout(**plotly_layout_defaults(title="Annual price volatility (CV)"), height=420)
        st.plotly_chart(fig_hm, width="stretch", key="brief_price_vol_heatmap")
        register("price_vol_heatmap", fig_hm)

    if not lcree_country.empty and not price_vol.empty:
        bc = lcree_country.copy()
        bc["year"] = pd.to_numeric(bc["year"], errors="coerce")
        acq = bc.loc[
            (bc["country"] == "United Kingdom")
            & bc["indicator"].astype(str).str.contains("Acquisitions", case=False, na=False),
            ["year", "value"],
        ].rename(columns={"value": "acquisitions_thousand_gbp"})
        gas = price_vol.loc[price_vol["series"] == "gas", ["year", "cv"]].rename(columns={"cv": "gas_cv"})
        acq["year"] = acq["year"].astype(int)
        gas["year"] = pd.to_numeric(gas["year"], errors="coerce").astype(int)
        sc = acq.merge(gas, on="year", how="inner")
        if not sc.empty:
            fig_sc = px.scatter(
                sc,
                x="gas_cv",
                y="acquisitions_thousand_gbp",
                hover_data=["year"],
                labels={"gas_cv": "Gas price CV", "acquisitions_thousand_gbp": "LCREE acquisitions (£k)"},
            )
            fig_sc.update_layout(**plotly_layout_defaults(), height=400)
            st.plotly_chart(fig_sc, width="stretch", key="brief_acq_vs_gas_vol")
            register("acq_vs_gas_vol", fig_sc)

    nd = queue_pipeline(non_dom)
    if not nd.empty:
        fig_nd = go.Figure()
        for col in ("received", "processed", "queue"):
            if col in nd.columns:
                fig_nd.add_trace(go.Scatter(x=nd["date"], y=nd[col], mode="lines+markers", name=col))
        fig_nd.update_layout(**plotly_layout_defaults(title="Non-domestic RHI pipeline"), height=380)
        st.plotly_chart(fig_nd, width="stretch", key="brief_nd_rhi_pipeline")
        register("nd_rhi_pipeline", fig_nd)

    sa = scheme_admin_actuals(scheme_adm)
    if not sa.empty:
        fig_sa = px.line(sa, x="date", y="pct", color="series", labels={"pct": "% of target"})
        fig_sa.update_layout(**plotly_layout_defaults(title="RHI scheme administration vs target"), height=360)
        st.plotly_chart(fig_sa, width="stretch", key="brief_scheme_admin")
        register("scheme_admin", fig_sa)

    if not lc250.empty:
        st.markdown("**LCREE 250+ businesses (raw wide sheet)** — inspect sector concentration.")
        st.dataframe(lc250.head(12), width="stretch", hide_index=True)

# ----- Tab 4 ----------------------------------------------------------------
with tabs[4]:
    st.subheader("Regional & structural — LCREE by country, biomass vs agriculture")
    if not lcree_country.empty:
        bc = lcree_country.copy()
        bc["year"] = pd.to_numeric(bc["year"], errors="coerce")
        bc = bc[bc["year"].between(yr[0], yr[1])]
        for ind in ("Employment (full time equivalent)", "Turnover (£ thousand)"):
            sub = bc.loc[bc["indicator"].astype(str).str.startswith(ind[:8]), ["country", "year", "value"]]
            if sub.empty:
                continue
            figc = px.line(
                sub.sort_values(["country", "year"]),
                x="year",
                y="value",
                color="country",
                labels={"value": ind},
            )
            figc.update_layout(**plotly_layout_defaults(title=ind), height=400)
            reg_lc = f"lcree_country_{ind[:5]}"
            st.plotly_chart(figc, width="stretch", key=f"brief_{reg_lc}")
            register(reg_lc, figc)

    if not renew09.empty and not fte_ind.empty:
        st.caption("Biomass proxy: sum of plant + animal biomass columns in sheet 09 (annual row).")
        bio_cols = [c for c in ("plant_biomass", "animal_biomass") if c in renew09.columns]
        if bio_cols and "industry" in renew09.columns:
            r9 = renew09.copy()
            r9["year"] = pd.to_numeric(r9["industry"], errors="coerce")
            r9 = r9.dropna(subset=["year"])
            r9["bio_mtoe"] = r9[bio_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            # Agriculture FTE from cleaned wide sheet is non-trivial; skip scatter if unreadable
            st.dataframe(r9[["year", "bio_mtoe"]].tail(8), hide_index=True, width="stretch")

# ----- Tab 5 ----------------------------------------------------------------
with tabs[5]:
    st.subheader("Integrated productivity (Section VI)")
    if integrated.empty:
        st.warning("Run `python3 scripts/build_derived_facts.py` for `integrated_productivity`.")
    else:
        st.caption(
            "LCREE coverage caveat — the 2024 release is missing the *Electricity, "
            "gas, steam and air conditioning supply* row in the published TO/FTE-by-"
            "industry tables, so the 2024 turnover-per-GWh and FTE-per-TWh values are "
            "**not directly comparable** with prior years.  Treat the latest point as "
            "a partial-coverage estimate."
        )
        ip = integrated.copy()
        ip["year"] = pd.to_numeric(ip["year"], errors="coerce")
        ip = ip[ip["year"].between(yr[0], yr[1])].sort_values("year")
        for col in (
            "turnover_per_gwh_thousand_gbp",
            "fte_per_twh",
            "acquisitions_per_new_mw_thousand_gbp",
        ):
            if col in ip.columns:
                ip[f"{col}_roll3"] = rolling_3y(pd.to_numeric(ip[col], errors="coerce")).values
        fig_ip = go.Figure()
        if "turnover_per_gwh_thousand_gbp" in ip.columns:
            fig_ip.add_trace(
                go.Scatter(x=ip["year"], y=ip["turnover_per_gwh_thousand_gbp"], name="£k / GWh", yaxis="y")
            )
        if "fte_per_twh" in ip.columns:
            fig_ip.add_trace(go.Scatter(x=ip["year"], y=ip["fte_per_twh"], name="FTE / TWh", yaxis="y2"))
        fig_ip.update_layout(
            **plotly_layout_defaults(),
            height=440,
            yaxis=dict(title="£k / GWh"),
            yaxis2=dict(title="FTE / TWh", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(fig_ip, width="stretch", key="brief_integrated_productivity")
        register("integrated_productivity", fig_ip)
        st.dataframe(ip.round(2), hide_index=True, width="stretch")

# ----- Tab 6 ----------------------------------------------------------------
with tabs[6]:
    st.subheader("Structural breaks — load factors & policy milestones")
    if not annual_lf.empty:
        for tech in ("Solar PV", "Wind", "Anaerobic Digestion"):
            sub = annual_lf.loc[annual_lf["technology"] == tech].copy()
            sub["fy_start"] = pd.to_numeric(sub["financial_year"].astype(str).str[:4], errors="coerce")
            sub = sub.dropna(subset=["fy_start", "weighted_mean"]).sort_values("fy_start")
            if len(sub) >= 8:
                arr = sub["weighted_mean"].astype(float).values
                rp = ruptures_peaks(arr, penalty=12.0)
                st.write(f"**{tech}** — PELT breakpoints (indices):", rp)
                ch = chow_mean_shift(
                    pd.Series(sub["weighted_mean"].values),
                    breakpoint_year=2016,
                    year_index=pd.Series(sub["fy_start"].values),
                )
                st.write("Welch mean-shift around FY 2016:", ch)
            fig_br = px.line(
                sub, x="fy_start", y="weighted_mean", markers=True, title=f"{tech} weighted mean LF"
            )
            fig_br.update_layout(**plotly_layout_defaults(), height=340)
            reg_br = f"breaks_{tech}"
            st.plotly_chart(fig_br, width="stretch", key=f"brief_{reg_br}")
            register(reg_br, fig_br)

    if not lcree.empty:
        mfg = lcree[lcree["sector"].astype(str).str.contains("Manufacturing", case=False, na=False)]
        mfg = mfg[mfg["year"].between(yr[0], yr[1])]
        if not mfg.empty:
            fig_m = px.line(
                mfg.sort_values("year"),
                x="year",
                y="fte",
                color="sector",
                labels={"fte": "LCREE FTE (manufacturing-related rows)"},
            )
            fig_m.update_layout(**plotly_layout_defaults(), height=380)
            st.plotly_chart(fig_m, width="stretch", key="brief_mfg_fte")
            register("mfg_fte", fig_m)

# ----- Tab 7 ----------------------------------------------------------------
with tabs[7]:
    st.subheader("Risk dashboard — heuristic flags")
    flags: list[str] = []
    if not annual_lf.empty:
        pv = annual_lf.loc[annual_lf["technology"] == "Solar PV"].copy()
        pv["fy_start"] = pd.to_numeric(pv["financial_year"].astype(str).str[:4], errors="coerce")
        pv = pv.sort_values("fy_start")
        if len(pv) >= 6:
            lf0 = float(pv["weighted_mean"].iloc[-5])
            lf1 = float(pv["weighted_mean"].iloc[-1])
            if lf0 - lf1 > 1.0:
                flags.append("Performance drift: Solar PV weighted LF fell >1 pp over ~5 fiscal years.")
    if not lcree.empty:
        sol = lcree[lcree["sector"].astype(str).str.contains("Solar", case=False, na=False)]
        if not sol.empty and "fte" in sol.columns:
            sol_y = sol.groupby("year", as_index=False)["fte"].sum().sort_values("year")
            if len(sol_y) >= 3:
                f0, f1 = float(sol_y["fte"].iloc[-3]), float(sol_y["fte"].iloc[-1])
                if f1 < f0 * 0.9:
                    flags.append("Cost–employment squeeze proxy: Solar-sector FTE trending down vs recent years.")
    if not price_m.empty:
        pm = coerce_monthly(price_m)
        rv = rolling_volatility(pm, "gas", window=12)
        if not rv.empty and rv["rolling_cv"].iloc[-1] > 0.5:
            flags.append("Price vulnerability: trailing-12m gas index CV elevated.")
    if not scheme_adm.empty:
        sa = scheme_admin_actuals(scheme_adm)
        if not sa.empty and (sa["series"] == "non_domestic_rhi").any():
            last_nd = sa.loc[sa["series"] == "non_domestic_rhi", "pct"].iloc[-1]
            if last_nd >= 99 and not queue_pipeline(non_dom).empty:
                q = queue_pipeline(non_dom)["queue"].iloc[-1]
                if q > 120:
                    flags.append(
                        "Regulatory risk: non-domestic RHI budget near target while queue remains elevated."
                    )
    if not flags:
        st.success("No heuristic flags triggered on current thresholds.")
    else:
        for f in flags:
            st.warning(f)

# ----- Tab 8 IRENA ----------------------------------------------------------
with tabs[8]:
    st.subheader("IRENA benchmarks — peers, CF heatmap, radar, CAGR ladder")
    if not sdg_uk.empty:
        fig_sdg = go.Figure()
        fig_sdg.add_trace(
            go.Scatter(x=sdg_uk["year"], y=sdg_uk["sdg_7b1_w_per_capita"], name="United Kingdom", mode="lines+markers")
        )
        if not sdg_oecd.empty:
            col = "oecd_benchmark" if "oecd_benchmark" in sdg_oecd.columns else sdg_oecd.columns[-1]
            fig_sdg.add_trace(
                go.Scatter(
                    x=sdg_oecd["year"],
                    y=sdg_oecd[col],
                    name="OECD aggregate",
                    mode="lines+markers",
                    line=dict(dash="dash"),
                )
            )
        fig_sdg.update_layout(**plotly_layout_defaults(), height=420, yaxis_title="W per inhabitant")
        st.plotly_chart(fig_sdg, width="stretch", key="brief_sdg_trajectory")
        register("sdg_trajectory", fig_sdg)

    cf_tbl_year = cf_tbl.copy()
    if not cf_tbl_year.empty:
        cf_tbl_year["year"] = pd.to_numeric(cf_tbl_year["year"], errors="coerce")
        cf_tbl_year = cf_tbl_year[cf_tbl_year["year"].between(yr[0], yr[1])]
    heat_tech = st.selectbox("Heatmap technology", tech_sel or tech_opts[:1])
    if heat_tech and not cf_tbl_year.empty:
        peer_names = [iso_to_name.get(i, i) for i in [UK_ISO3, *peer_iso3]]
        top_y = int(cf_tbl_year["year"].max())
        top5 = top_n_countries_by_cf(cf_tbl, heat_tech, top_y, n=5)
        extra = top5["country"].tolist() if not top5.empty else []
        hm_countries = sorted(set(peer_names + extra))
        hm = cf_tbl_year.loc[
            (cf_tbl_year["technology"] == heat_tech) & (cf_tbl_year["country"].isin(hm_countries))
        ]
        pivot = hm.pivot_table(index="country", columns="year", values="capacity_factor")
        pivot_z = pivot.astype("float64")
        zhi = pivot_z.max(axis=None, skipna=True)
        zmax = min(1.0, float(zhi)) if pd.notna(zhi) else 1.0
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=pivot_z.values,
                x=pivot_z.columns.astype(str),
                y=pivot_z.index.astype(str),
                colorscale="Viridis",
                zmin=0,
                zmax=zmax,
            )
        )
        fig_hm.update_layout(**plotly_layout_defaults(title=f"CF heatmap — {heat_tech}"), height=480)
        st.plotly_chart(fig_hm, width="stretch", key="brief_cf_heatmap")
        register("cf_heatmap", fig_hm)

    last_common = int(cf_tbl_year["year"].max()) if not cf_tbl_year.empty else None
    if last_common is not None:
        snap = cf_tbl_year.loc[cf_tbl_year["year"] == last_common]
        med = snap.groupby("technology")["capacity_factor"].median()
        uk_snap = snap.loc[snap["iso3_code"] == UK_ISO3]
        uk_row = uk_snap.groupby("technology")["capacity_factor"].mean()
        techs_r = [t for t in tech_sel if t in med.index and t in uk_row.index]
        if techs_r:
            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(r=uk_row.loc[techs_r].values, theta=techs_r, fill="toself", name="UK")
            )
            fig_radar.add_trace(
                go.Scatterpolar(r=med.loc[techs_r].values, theta=techs_r, fill="toself", name="Global median")
            )
            fig_radar.update_layout(
                **plotly_layout_defaults(),
                height=460,
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            )
            st.plotly_chart(fig_radar, width="stretch", key="brief_radar_cf")
            register("radar_cf", fig_radar)

    ladder_rows: list[dict] = []
    for t in tech_sel or uk_fact_agg["technology"].dropna().unique():
        for period_label, a, b in (
            (f"{pre_lo}–{pre_hi}", pre_lo, pre_hi),
            (f"{post_lo}–{post_hi}", post_lo, post_hi),
        ):
            g = uk_fact_agg.loc[(uk_fact_agg["technology"] == t) & uk_fact_agg["year"].between(a, b)].sort_values(
                "year"
            )
            if len(g) < 2:
                continue
            cap_s = float(g["electricity_installed_capacity_mw"].iloc[0])
            cap_e = float(g["electricity_installed_capacity_mw"].iloc[-1])
            gen_s = float(g["electricity_generation_gwh"].iloc[0])
            gen_e = float(g["electricity_generation_gwh"].iloc[-1])
            cf_s = float(g["cf"].iloc[0])
            cf_e = float(g["cf"].iloc[-1])
            ny = float(g["year"].iloc[-1] - g["year"].iloc[0])
            if ny <= 0:
                continue
            ladder_rows.append(
                {
                    "technology": t,
                    "period": period_label,
                    "CAGR capacity": cagr(cap_s, cap_e, ny),
                    "CAGR generation": cagr(gen_s, gen_e, ny),
                    "CAGR CF": cagr(cf_s, cf_e, ny),
                }
            )
    if ladder_rows:
        ladder_df = pd.DataFrame(ladder_rows)
        st.dataframe(ladder_df.round(4), width="stretch", hide_index=True)

# ----- Tab 9 ----------------------------------------------------------------
with tabs[9]:
    st.subheader("Visualization annex")
    for name, fig in sf.items():
        st.markdown(f"**{name}**")
        st.plotly_chart(fig, width="stretch", key=f"annex_{name}")
    st.divider()
    st.markdown(
        """
### Methodology (Section X)

- **Renewable share of electricity:** annual sum of DESNZ quarterly TWh;
  renewables = wind+solar + hydro + bioenergy; share = renewables / total
  generation including interconnectors.
- **Industry renewable share:** sheet 09 renewable Mtoe by industry ×
  41.868 → PJ, divided by sheet 15 total PJ (first industry block).
- **RHI unit economics:** year-end cumulative payments (£m) and installs;
  annual new = diff; cost per install = Δ£m × 1e6 / Δinstalls.  Chart uses
  2024-deflated £ when `config/reference/cpi_uk.csv` exists.
- **Solar learning:** log-log regression of median £/kW on cumulative UK PV
  MW; learning rate = 1 − 2^slope.  Deflate with CPI when
  `config/reference/cpi_uk.csv` exists.
- **Load factors:** DESNZ FIT weighted means; note **coverage** field.
- **Acquisitions / new MW:** LCREE UK acquisitions (£k) ÷ first-difference
  of summed IRENA UK renewable capacity (MW); zero-denominator years are
  omitted.

### Strategic questions (Section XI)

Is low-carbon economic growth matched by physical efficiency and domestic
value, or by falling costs and import-heavy deployment?  Use the tabs above
together — not any single chart in isolation.
"""
    )
