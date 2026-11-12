"""The Recall step's chart: detections to check (real and false, stacked)
against the recall aimed for, the likely range dashed, the chosen target marked."""
import altair as alt
import polars as pl


def cost_chart(curve: pl.DataFrame, target: float, kept: int, col: dict, styled: bool = True) -> alt.Chart:
    long = curve.unpivot(index=["recall", "kept", "hours", "lo", "hi"], on=["real", "false"],
                         variable_name="part", value_name="n").with_columns(
        part=pl.col("part").replace({"real": "Real calls", "false": "False detections"}))
    x = alt.X("recall:Q", title="Recall (share of real calls kept)", axis=alt.Axis(format="%"),
              scale=alt.Scale(domain=[0.3, 1.0]))
    tip = [alt.Tooltip("recall:Q", format=".0%", title="Recall"), alt.Tooltip("kept:Q", format=",", title="To check"),
           alt.Tooltip("hours:N", title="Volunteer time")]
    area = alt.Chart(long.to_pandas()).mark_area(opacity=0.85).encode(
        x=x, y=alt.Y("n:Q", stack=True, title="Detections to check"),
        color=alt.Color("part:N", title=None, sort=["Real calls", "False detections"],
                        scale=alt.Scale(domain=["Real calls", "False detections"], range=[col["yes"], col["no"]]),
                        legend=alt.Legend(orient="top-left")),
        order=alt.Order("part:N", sort="descending"), tooltip=tip)
    edges = alt.Chart(curve.to_pandas()).transform_fold(["lo", "hi"]).mark_line(
        strokeDash=[4, 3], strokeWidth=1.2, color=col["ink"], opacity=0.55).encode(x=x, y="value:Q", detail="key:N")
    rule = alt.Chart(pl.DataFrame({"recall": [target]}).to_pandas()).mark_rule(
        color=col["accent"], strokeWidth=2).encode(x="recall:Q")
    dot = alt.Chart(pl.DataFrame([{"recall": target, "kept": kept}]).to_pandas()).mark_point(
        size=110, filled=True, color=col["accent"]).encode(x="recall:Q", y="kept:Q")
    chart = (area + edges + rule + dot).properties(height=340)
    if styled:
        from ui.common import style_chart
        chart = style_chart(chart)
    return chart
