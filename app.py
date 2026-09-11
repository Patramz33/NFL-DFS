import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

st.set_page_config(page_title="NFL DFS Dashboard V2", layout="wide")

DK_SALARY_CAP = 50000
ROSTER_SIZE = 9
REQUIRED_COLS = [
    "name", "position", "team", "opponent", "salary",
    "projection", "ceiling", "ownership", "status", "game"
]

WEEK1_MAIN_GAMES = [
    {"game":"CHI@CAR","away":"CHI","home":"CAR","kickoff":"1:00 PM ET"},
    {"game":"BAL@IND","away":"BAL","home":"IND","kickoff":"1:00 PM ET"},
    {"game":"ATL@PIT","away":"ATL","home":"PIT","kickoff":"1:00 PM ET"},
    {"game":"CLE@JAC","away":"CLE","home":"JAC","kickoff":"1:00 PM ET"},
    {"game":"TB@CIN","away":"TB","home":"CIN","kickoff":"1:00 PM ET"},
    {"game":"NYJ@TEN","away":"NYJ","home":"TEN","kickoff":"1:00 PM ET"},
    {"game":"NO@DET","away":"NO","home":"DET","kickoff":"1:00 PM ET"},
    {"game":"BUF@HOU","away":"BUF","home":"HOU","kickoff":"1:00 PM ET"},
    {"game":"ARI@LAC","away":"ARI","home":"LAC","kickoff":"4:25 PM ET"},
    {"game":"GB@MIN","away":"GB","home":"MIN","kickoff":"4:25 PM ET"},
    {"game":"MIA@LV","away":"MIA","home":"LV","kickoff":"4:25 PM ET"},
    {"game":"WAS@PHI","away":"WAS","home":"PHI","kickoff":"4:25 PM ET"},
]

TEAM_TO_GAME = {}
TEAM_TO_OPP = {}
TEAM_TO_KICKOFF = {}
for g in WEEK1_MAIN_GAMES:
    TEAM_TO_GAME[g["away"]] = g["game"]
    TEAM_TO_GAME[g["home"]] = g["game"]
    TEAM_TO_OPP[g["away"]] = g["home"]
    TEAM_TO_OPP[g["home"]] = g["away"]
    TEAM_TO_KICKOFF[g["away"]] = g["kickoff"]
    TEAM_TO_KICKOFF[g["home"]] = g["kickoff"]


def normalize_df(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename = {
        "avgpointspergame": "projection",
        "avg_points_per_game": "projection",
        "name_+_id": "name",
        "roster_position": "position",
        "teamabbrev": "team",
        "team_abbrev": "team",
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

    for col in ["salary","projection","ceiling","ownership"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["name","position","team","opponent","status","game"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    if "position" in df.columns:
        df["position"] = df["position"].str.upper().replace({"D":"DST","DEF":"DST"})
        df["position"] = df["position"].str.split("/").str[0]

    if "team" in df.columns:
        df["team"] = df["team"].str.upper().replace({"JAX":"JAC","WSH":"WAS","LAR":"LA","OAK":"LV"})

    # Fill slate metadata from team if omitted.
    if "opponent" not in df.columns:
        df["opponent"] = df.get("team", pd.Series(index=df.index, dtype=str)).map(TEAM_TO_OPP).fillna("")
    else:
        missing = df["opponent"].eq("")
        df.loc[missing, "opponent"] = df.loc[missing, "team"].map(TEAM_TO_OPP).fillna("")

    if "game" not in df.columns:
        df["game"] = df.get("team", pd.Series(index=df.index, dtype=str)).map(TEAM_TO_GAME).fillna("")
    else:
        missing = df["game"].eq("")
        df.loc[missing, "game"] = df.loc[missing, "team"].map(TEAM_TO_GAME).fillna("")

    if "status" not in df.columns:
        df["status"] = ""

    if "ownership" in df.columns:
        mask = df["ownership"].between(0, 1, inclusive="both")
        df.loc[mask, "ownership"] = df.loc[mask, "ownership"] * 100

    if "salary" in df.columns and "projection" in df.columns:
        df["value"] = np.where(df["salary"] > 0, df["projection"] / df["salary"] * 1000, np.nan)
    if "ceiling" in df.columns and "ownership" in df.columns:
        own = df["ownership"].clip(lower=0.5)
        df["leverage"] = df["ceiling"] / own

    if "kickoff" not in df.columns:
        df["kickoff"] = df.get("team", pd.Series(index=df.index, dtype=str)).map(TEAM_TO_KICKOFF).fillna("")

    return df


def validate(df):
    return [c for c in REQUIRED_COLS if c not in df.columns]


def build_optimization(player_df, objective_col, locked_names, excluded_names,
                       min_qb_stack=0, require_bringback=False, max_team_players=4,
                       min_salary=0, max_ownership=None, no_good_lineups=None,
                       min_unique=2, exposure_caps=None, lineup_index=0,
                       game_limits=None):
    df = player_df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        return None, "No players available after filtering."

    c = -df[objective_col].fillna(0).to_numpy(dtype=float)
    rows, lb, ub = [], [], []

    def add(coeff, low, high):
        rows.append(np.asarray(coeff, dtype=float)); lb.append(low); ub.append(high)

    add(np.ones(n), ROSTER_SIZE, ROSTER_SIZE)
    sal = df["salary"].to_numpy(dtype=float)
    add(sal, min_salary, DK_SALARY_CAP)

    pos = df["position"].to_numpy()
    add((pos=="QB").astype(float), 1, 1)
    add((pos=="DST").astype(float), 1, 1)
    add((pos=="TE").astype(float), 1, 2)
    add((pos=="RB").astype(float), 2, 3)
    add((pos=="WR").astype(float), 3, 4)
    add(np.isin(pos, ["RB","WR","TE"]).astype(float), 7, 7)

    if max_ownership is not None:
        add(df["ownership"].fillna(0).to_numpy(float), 0, max_ownership)

    for i, name in enumerate(df["name"]):
        if name in locked_names:
            coeff = np.zeros(n); coeff[i] = 1; add(coeff, 1, 1)
        if name in excluded_names:
            coeff = np.zeros(n); coeff[i] = 1; add(coeff, 0, 0)

    for team in sorted(df["team"].dropna().unique()):
        coeff = (df["team"].to_numpy() == team).astype(float)
        add(coeff, 0, max_team_players)

    # Optional game exposure limits, e.g. max 4 players from any one game.
    if game_limits:
        max_game_players = game_limits.get("max_per_game")
        if max_game_players:
            for game in sorted(df["game"].dropna().unique()):
                if not game:
                    continue
                coeff = (df["game"].to_numpy() == game).astype(float)
                add(coeff, 0, max_game_players)

    if min_qb_stack > 0:
        for qi in df.index[df["position"]=="QB"]:
            team = df.at[qi, "team"]
            coeff = (df["team"].eq(team) & df["position"].isin(["WR","TE"])).to_numpy(dtype=float)
            coeff[qi] -= min_qb_stack
            add(coeff, 0, np.inf)

    if require_bringback:
        for qi in df.index[df["position"]=="QB"]:
            opp = df.at[qi, "opponent"]
            coeff = (df["team"].eq(opp) & df["position"].isin(["RB","WR","TE"])).to_numpy(dtype=float)
            coeff[qi] -= 1
            add(coeff, 0, np.inf)

    # Uniqueness from previously generated lineups.
    if no_good_lineups:
        for prev_names in no_good_lineups:
            coeff = df["name"].isin(prev_names).to_numpy(dtype=float)
            add(coeff, 0, ROSTER_SIZE - min_unique)

    # Exposure caps are translated into temporary exclusions when cap is reached.
    if exposure_caps:
        for i, name in enumerate(df["name"]):
            cap_count = exposure_caps.get(name)
            if cap_count is not None and cap_count <= lineup_index:
                # This branch is intentionally not used directly; exclusions are computed outside.
                pass

    A = csr_matrix(np.vstack(rows))
    cons = LinearConstraint(A, np.array(lb, dtype=float), np.array(ub, dtype=float))
    result = milp(
        c=c,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=cons,
        options={"time_limit": 15}
    )

    if not result.success or result.x is None:
        return None, f"Optimizer could not find a valid lineup. {result.message}"
    return df[result.x > 0.5].copy(), None


def assign_slots(lineup):
    lineup = lineup.copy()
    rows = []
    qb = lineup[lineup.position=="QB"]
    dst = lineup[lineup.position=="DST"]
    rb = lineup[lineup.position=="RB"].sort_values("projection", ascending=False)
    wr = lineup[lineup.position=="WR"].sort_values("projection", ascending=False)
    te = lineup[lineup.position=="TE"].sort_values("projection", ascending=False)

    used = set()
    if len(qb): rows.append(("QB", qb.index[0])); used.add(qb.index[0])
    for idx in rb.index[:2]: rows.append(("RB", idx)); used.add(idx)
    for idx in wr.index[:3]: rows.append(("WR", idx)); used.add(idx)
    if len(te): rows.append(("TE", te.index[0])); used.add(te.index[0])
    remaining = lineup[lineup.position.isin(["RB","WR","TE"]) & ~lineup.index.isin(used)]
    if len(remaining):
        flex_idx = remaining.sort_values("projection", ascending=False).index[0]
        rows.append(("FLEX", flex_idx)); used.add(flex_idx)
    if len(dst): rows.append(("DST", dst.index[0]))

    out = []
    for slot, idx in rows:
        r = lineup.loc[idx]
        out.append({
            "Slot": slot, "Player": r["name"], "Pos": r["position"],
            "Team": r["team"], "Opp": r["opponent"], "Game": r["game"],
            "Kickoff": r.get("kickoff", ""), "Salary": int(r["salary"]),
            "Projection": round(float(r["projection"]),2),
            "Ceiling": round(float(r["ceiling"]),2),
            "Own %": round(float(r["ownership"]),1),
            "Value": round(float(r["value"]),2),
        })
    return pd.DataFrame(out)


def generate_portfolio(df, num_lineups, objective, locked, excluded, min_qb_stack,
                       bringback, max_team, min_salary, max_lineup_ownership,
                       min_unique, default_max_exposure, custom_exposure, max_per_game):
    lineups = []
    raw_lineups = []
    exposure_counts = {name: 0 for name in df["name"]}

    max_counts = {}
    for name in df["name"]:
        pct = custom_exposure.get(name, default_max_exposure)
        max_counts[name] = max(1, math.floor(num_lineups * pct / 100.0 + 1e-9))

    for li in range(num_lineups):
        dynamic_excluded = set(excluded)
        for name, count in exposure_counts.items():
            if count >= max_counts.get(name, num_lineups):
                dynamic_excluded.add(name)

        lineup, err = build_optimization(
            df,
            objective,
            set(locked),
            dynamic_excluded,
            min_qb_stack=min_qb_stack,
            require_bringback=bringback,
            max_team_players=max_team,
            min_salary=min_salary,
            max_ownership=max_lineup_ownership,
            no_good_lineups=[set(x["name"]) for x in raw_lineups],
            min_unique=min_unique,
            lineup_index=li,
            game_limits={"max_per_game": max_per_game},
        )
        if err:
            return lineups, raw_lineups, exposure_counts, f"Stopped after {li} lineup(s): {err}"

        raw_lineups.append(lineup)
        lineups.append(assign_slots(lineup))
        for name in lineup["name"]:
            exposure_counts[name] += 1

    return lineups, raw_lineups, exposure_counts, None


st.title("🏈 NFL DFS Dashboard — Version 2")
st.caption("DraftKings Classic • 2026 Week 1 Main Slate • 12 games • Multi-lineup portfolio tools")

with st.sidebar:
    st.header("Slate setup")
    uploaded = st.file_uploader("Upload full player pool CSV", type=["csv"])
    use_sample = st.checkbox("Use built-in 12-game test pool", value=(uploaded is None))
    st.divider()
    st.header("Optimizer")
    objective = st.selectbox("Optimize for", ["projection","ceiling","gpp_score"])
    min_qb_stack = st.selectbox("QB stack", [0,1,2], index=1,
                                format_func=lambda x: "None" if x==0 else f"QB + {x} WR/TE")
    bringback = st.checkbox("Require opponent bring-back", value=False)
    max_team = st.slider("Max players from one NFL team", 3, 6, 4)
    max_per_game = st.slider("Max players from one game", 3, 7, 5)
    min_salary = st.slider("Minimum salary spend", 0, 50000, 48000, step=500)
    max_lineup_ownership = st.number_input("Max combined ownership % (0 = no cap)", 0.0, 900.0, 0.0, 5.0)
    st.divider()
    st.header("Portfolio")
    num_lineups = st.slider("Number of lineups", 1, 50, 10)
    min_unique = st.slider("Minimum unique players between lineups", 1, 5, 2)
    default_max_exposure = st.slider("Default max player exposure %", 10, 100, 70, 5)
    st.divider()
    st.caption("Decision-support only. V2 does not scrape or submit entries to DraftKings.")

@st.cache_data
def load_sample():
    return pd.read_csv(Path(__file__).with_name("sample_week1_main_slate.csv"))

if uploaded is not None:
    raw = pd.read_csv(uploaded)
elif use_sample:
    raw = load_sample()
else:
    st.info("Upload a CSV or enable the built-in test pool.")
    st.stop()

df = normalize_df(raw)
missing = validate(df)
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

# Restrict to the 12-game main slate when teams are recognizable.
main_teams = set(TEAM_TO_GAME)
recognized = df["team"].isin(main_teams)
if recognized.any():
    df = df[recognized].copy()

# Derivatives.
df["gpp_score"] = (
    0.50 * df["projection"].fillna(0)
    + 0.35 * df["ceiling"].fillna(0)
    + 0.10 * (df["ceiling"].fillna(0) / df["ownership"].clip(lower=1))
    + 0.05 * df["value"].fillna(0)
)

active_df = df[~df["status"].str.upper().isin(["OUT","IR"])].copy()

m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("Players", len(df))
m2.metric("Games", df["game"].nunique())
m3.metric("Avg Projection", f"{df['projection'].mean():.1f}")
m4.metric("Avg Ownership", f"{df['ownership'].mean():.1f}%")
m5.metric("Best Value", f"{df['value'].max():.2f}x")
m6.metric("Salary Cap", "$50,000")

schedule_df = pd.DataFrame(WEEK1_MAIN_GAMES)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Slate Overview", "Player Pool", "Lineup Builder", "Portfolio", "CSV Guide"
])

with tab1:
    st.subheader("2026 Week 1 Main Slate")
    st.dataframe(schedule_df, use_container_width=True, hide_index=True)

    st.subheader("Game environment from loaded pool")
    game_summary = (
        df.groupby(["game","kickoff"], dropna=False)
        .agg(players=("name","count"),
             avg_projection=("projection","mean"),
             avg_ceiling=("ceiling","mean"),
             total_projected_ownership=("ownership","sum"))
        .reset_index()
        .sort_values(["kickoff","avg_ceiling"], ascending=[True,False])
    )
    st.dataframe(game_summary, use_container_width=True, hide_index=True)

with tab2:
    c1,c2,c3,c4,c5 = st.columns(5)
    positions = c1.multiselect("Position", ["QB","RB","WR","TE","DST"], default=["QB","RB","WR","TE","DST"])
    teams = c2.multiselect("Team", sorted(df.team.unique()), default=[])
    games = c3.multiselect("Game", [g["game"] for g in WEEK1_MAIN_GAMES], default=[])
    max_own = c4.slider("Max ownership %", 0.0, float(max(50, int(df.ownership.max()+5))), float(max(50, int(df.ownership.max()+5))))
    min_proj = c5.slider("Min projection", 0.0, float(max(40, int(df.projection.max()+5))), 0.0)

    pool = df[df.position.isin(positions)].copy()
    if teams: pool = pool[pool.team.isin(teams)]
    if games: pool = pool[pool.game.isin(games)]
    pool = pool[(pool.ownership <= max_own) & (pool.projection >= min_proj)]

    cols = ["name","position","team","opponent","game","kickoff","salary","projection","ceiling","ownership","value","leverage","status"]
    st.dataframe(pool[cols].sort_values(["position","projection"], ascending=[True,False]), use_container_width=True, hide_index=True)

    a,b = st.columns(2)
    with a:
        st.subheader("Projection vs Salary")
        st.scatter_chart(pool[["salary","projection","position"]].dropna(), x="salary", y="projection", color="position", use_container_width=True)
    with b:
        st.subheader("Ceiling vs Ownership")
        st.scatter_chart(pool[["ownership","ceiling","position"]].dropna(), x="ownership", y="ceiling", color="position", use_container_width=True)

with tab3:
    st.subheader("Single-Lineup Builder")
    selectable = sorted(active_df.name.tolist())
    a,b = st.columns(2)
    locked = a.multiselect("Lock players", selectable, key="single_locks")
    excluded = b.multiselect("Exclude players", [n for n in selectable if n not in locked], key="single_excludes")

    if st.button("⚙️ Optimize single lineup", type="primary"):
        lineup, err = build_optimization(
            active_df,
            objective, set(locked), set(excluded),
            min_qb_stack=min_qb_stack,
            require_bringback=bringback,
            max_team_players=max_team,
            min_salary=min_salary,
            max_ownership=(max_lineup_ownership if max_lineup_ownership > 0 else None),
            game_limits={"max_per_game": max_per_game},
        )
        if err:
            st.error(err)
        else:
            st.session_state["last_lineup_v2"] = assign_slots(lineup)
            st.session_state["last_raw_lineup_v2"] = lineup

    if "last_lineup_v2" in st.session_state:
        roster = st.session_state["last_lineup_v2"]
        raw_lineup = st.session_state["last_raw_lineup_v2"]
        a,b,c,d = st.columns(4)
        a.metric("Salary", f"${roster['Salary'].sum():,.0f}")
        b.metric("Projection", f"{roster['Projection'].sum():.2f}")
        c.metric("Ceiling", f"{roster['Ceiling'].sum():.2f}")
        d.metric("Combined Own.", f"{roster['Own %'].sum():.1f}%")
        st.dataframe(roster, use_container_width=True, hide_index=True)
        st.download_button("Download lineup CSV", roster.to_csv(index=False).encode("utf-8"), "optimized_lineup_v2.csv", "text/csv")

with tab4:
    st.subheader("Multi-Lineup Portfolio")
    selectable = sorted(active_df.name.tolist())
    a,b = st.columns(2)
    locks = a.multiselect("Portfolio locks", selectable, key="portfolio_locks")
    excludes = b.multiselect("Portfolio exclusions", [n for n in selectable if n not in locks], key="portfolio_excludes")

    st.markdown("**Optional custom max exposure overrides**")
    exposure_players = st.multiselect("Players with custom exposure", selectable, key="exp_players")
    custom_exposure = {}
    if exposure_players:
        exp_cols = st.columns(min(4, len(exposure_players)))
        for j, name in enumerate(exposure_players):
            custom_exposure[name] = exp_cols[j % len(exp_cols)].slider(name, 0, 100, default_max_exposure, 5, key=f"exp_{name}")

    if st.button("🧮 Generate portfolio", type="primary"):
        lineups, raw_lineups, exposure_counts, err = generate_portfolio(
            active_df, num_lineups, objective, locks, excludes, min_qb_stack,
            bringback, max_team, min_salary,
            (max_lineup_ownership if max_lineup_ownership > 0 else None),
            min_unique, default_max_exposure, custom_exposure, max_per_game
        )
        st.session_state["portfolio_lineups"] = lineups
        st.session_state["portfolio_raw"] = raw_lineups
        st.session_state["portfolio_exposure"] = exposure_counts
        st.session_state["portfolio_error"] = err

    if "portfolio_lineups" in st.session_state:
        lineups = st.session_state["portfolio_lineups"]
        raw_lineups = st.session_state["portfolio_raw"]
        err = st.session_state.get("portfolio_error")
        if err:
            st.warning(err)
        if lineups:
            summary_rows = []
            export_rows = []
            for i, roster in enumerate(lineups, start=1):
                summary_rows.append({
                    "Lineup": i,
                    "Salary": roster["Salary"].sum(),
                    "Projection": roster["Projection"].sum(),
                    "Ceiling": roster["Ceiling"].sum(),
                    "Combined Own %": roster["Own %"].sum(),
                })
                for _, r in roster.iterrows():
                    export_rows.append({"Lineup": i, **r.to_dict()})
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            exp = []
            for name in active_df["name"]:
                count = sum(name in set(x["name"]) for x in raw_lineups)
                if count:
                    exp.append({"Player": name, "Lineups": count, "Exposure %": round(100*count/len(raw_lineups),1)})
            exp_df = pd.DataFrame(exp).sort_values(["Exposure %","Player"], ascending=[False,True]) if exp else pd.DataFrame()
            st.subheader("Portfolio Exposure")
            st.dataframe(exp_df, use_container_width=True, hide_index=True)

            export_df = pd.DataFrame(export_rows)
            st.download_button("Download full portfolio CSV", export_df.to_csv(index=False).encode("utf-8"), "dfs_portfolio_v2.csv", "text/csv")

            choice = st.selectbox("Inspect lineup", list(range(1, len(lineups)+1)))
            st.dataframe(lineups[choice-1], use_container_width=True, hide_index=True)

with tab5:
    st.markdown("""
### Required columns
`name, position, team, opponent, salary, projection, ceiling, ownership, status, game`

For Week 1, V2 can automatically infer `opponent`, `game`, and `kickoff` from the team abbreviation when those fields are blank. Ownership can be entered as `18.5` or `0.185`.

### DraftKings salary-file workflow
1. Download the salary CSV manually from your contest/slate.
2. Add or merge `projection`, `ceiling`, and `ownership` from your preferred source.
3. Rename columns to the required format if necessary.
4. Upload the completed file here.

### What V2 adds
- Full 12-game Week 1 main-slate map
- Game and kickoff filtering
- Multi-lineup generation
- Max player exposure controls
- Custom player exposure overrides
- Lineup uniqueness constraints
- Max players per game
- Combined ownership cap
- Portfolio exposure report and CSV export

The included player pool is synthetic and for software testing only; it is **not** a current salary/projection feed.
""")
