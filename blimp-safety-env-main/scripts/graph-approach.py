import json
from pathlib import Path

import click
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@click.command()
@click.argument("params_file", type=click.Path(path_type=Path))
@click.argument("data_files", type=click.Path(path_type=Path), nargs=-1)
def main(params_file, data_files):
    """Process data."""
    with open(params_file) as fh:
        params = json.load(fh)
    dfs = []
    for data_file in data_files:
        df = pd.read_parquet(data_file)
        df = summarize(df)
        position(df, params)
        print(data_file.name)
        print(f"{df.num_balloons_collected.sum():0.3f} "
              f"({df.num_balloons_collected.std():0.3f})")
        print("")
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = summarize(df)
    graph(df)


def summarize(df):
    output = []
    steps = int(df.step.max())
    for idx in range(steps + 1):
        # data = df.query(f"step == {idx} and num_balloons_collected == 0")
        data = df.query(f"step == {idx}")
        if len(data):
            output.append(data.mean())
    return pd.concat(output, axis=1).T


def position(df, pos):
    df["position"] = 0.0
    for idx in range(len(df)):
        # df.position[idx] = pos[int(df.step[idx])]["fire_locations"][0][0]
        df.position[idx] = pos[int(df.step[idx])]["balloon_locations"][0][1]

def graph(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    scatter = go.Scatter(
        x=df.position,
        y=df.num_balloons_collected,
        mode='lines+markers',
        name="Balloons Collected",
    )
    bar = go.Bar(
        x=df.position,
        y=df.step_num,
        name="Steps",
    )
    fig.add_trace(scatter, row=1, col=1)
    fig.add_trace(bar, row=2, col=1)
    fig.update_layout(title_text="Average Model Performance")
    # fig["layout"]["xaxis2"]["title"] = "🔥 distance. 🎈 at 12, ✈️ at 0"
    fig["layout"]["xaxis2"]["title"] = "🎈 y position, 🔥 at 18. , ✈️ at 0"
    # fig["layout"]["xaxis2"]["title"] = "🎈 position at (12, y), 🔥 at (12, 0). , ✈️ at (0, 0)"
    fig["layout"]["yaxis"]["title"] = "Avg. Balloons collected"
    fig["layout"]["yaxis2"]["title"] = "Avg. Steps"
    fig.show()
    fig.write_image("image.png")


if __name__ == "__main__":
    main()
