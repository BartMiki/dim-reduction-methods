import os

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.decomposition import PCA
from umap import UMAP

from datasets import get_datasets

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.GRID]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
datasets = get_datasets()

app.layout = html.Div([
    html.H2('Compare different dimensionality reduction methods'),
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[dict(label=k, value=k) for k in datasets.keys()],
        value=next(iter(datasets.keys()))
    ),

    dbc.Row(
        [
            dbc.Col(html.Div([
                html.H4("Linear: MDS"),
                dcc.Graph(id='linear-mds')
            ]), width=4),
            dbc.Col(html.Div([
                html.H4(id='tsne-1-header'),
                dcc.Graph(id='tsne-1'),
                dcc.Slider(id='tsne-1-slider', min=5, max=50, step=1, value=30,
                           marks={str(i): str(i) for i in range(5, 51, 5)})
            ]), width=4),
            dbc.Col(html.Div([
                html.H4(id='umap-header'),
                dcc.Graph(id='umap'),
                dcc.Slider(id='umap-slider', min=2, max=29, step=1, value=15,
                           marks={str(i): str(i) for i in range(2, 30, 3)})
            ]), width=4),
        ]
    ),
    dbc.Row(
        [
            dbc.Col(html.Div([
                html.H4("Linear: PCA"),
                dcc.Graph(id='linear-pca')
            ]), width=4),
            dbc.Col(html.Div([
                html.H4(id='tsne-2-header'),
                dcc.Graph(id='tsne-2'),
                dcc.Slider(id='tsne-2-slider', min=5, max=50, step=1, value=10,
                           marks={str(i): str(i) for i in range(5, 51, 5)})
            ]), width=4),
            dbc.Col(html.Div([
                html.H4(id='isomap-header'),
                dcc.Graph(id='isomap'),
                dcc.Slider(id='isomap-slider', min=1, max=15, step=1, value=5,
                           marks={str(i): str(i) for i in range(1, 16, 2)})
            ]), width=4),
        ]
    ),
], style={'margin-left': '5%', 'margin-right': '5%'})


def render_figure(transformer, dataset_key):
    dataset = datasets[dataset_key]
    df = dataset.df
    df_hat = transformer.fit_transform(df.values)
    df_hat = pd.DataFrame(df_hat)
    df_hat.columns = ['x', 'y']

    fig = px.scatter(df_hat, x='x', y='y',
                     color=df.index if dataset.color_index else None,
                     hover_name=df.index if dataset.display_index else None)
    fig.update_traces(marker=dict(size=8))
    return fig


@app.callback(
    Output("linear-mds", "figure"),
    Input("dataset-dropdown", "value"))
def update_linear_mds(key):
    return render_figure(MDS(), key)


@app.callback(
    Output("linear-pca", "figure"),
    Input("dataset-dropdown", "value"))
def update_linear_pca(key):
    return render_figure(PCA(n_components=2), key)


@app.callback(
    Output("tsne-1", "figure"), Output("tsne-1-header", "children"),
    Input("dataset-dropdown", "value"), Input("tsne-1-slider", "value"))
def update_tsne_1(key, perplexity):
    return render_figure(TSNE(perplexity=int(perplexity)), key), f"TNSE with perplexity = {perplexity}"


@app.callback(
    Output("tsne-2", "figure"), Output("tsne-2-header", "children"),
    Input("dataset-dropdown", "value"), Input("tsne-2-slider", "value"))
def update_tsne_2(key, perplexity):
    return render_figure(TSNE(perplexity=int(perplexity)), key), f"TNSE with perplexity = {perplexity}"


@app.callback(
    Output("umap", "figure"), Output("umap-header", "children"),
    Input("dataset-dropdown", "value"), Input("umap-slider", "value"))
def update_umap(key, n_neighbors):
    return render_figure(UMAP(n_neighbors=int(n_neighbors)), key), f"UMAP with {n_neighbors} neighbors"


@app.callback(
    Output("isomap", "figure"), Output("isomap-header", "children"),
    Input("dataset-dropdown", "value"), Input("isomap-slider", "value"))
def update_isomap(key, n_neighbors):
    dataset = datasets[key]
    df = dataset.df
    n_neighbors = min(df.shape[0], int(n_neighbors))
    return render_figure(Isomap(n_neighbors=n_neighbors), key), f"Isomap with {n_neighbors} neighbors"


if __name__ == '__main__':
    app.run_server(debug=False)
