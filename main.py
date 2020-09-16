import dash
import dash_core_components as dcc
import dash_html_components as html
import glob
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash.dependencies import Input, Output, State

count = 0
def find_dataset():
    dataset_list = glob.glob('./dataset/*.txt')
    dict_dataset_list = []
    for idx, dataset in enumerate(dataset_list):
        dict_dataset_list.append({'label': dataset, 'value': idx})
    return dict_dataset_list

def read_data_from_txt(path):
    f = open(path, "r")
    data = f.read()
    data = data.replace('\n', '')
    set_of_data = data.split(';')
    voltage_set = []
    current_set = []
    for count in range(len(set_of_data) - 1):
        voltage_and_current = set_of_data[count].split(',')
        voltage_set.append(voltage_and_current[0])
        current_set.append(voltage_and_current[1])
    voltage_set = np.array(voltage_set).astype(np.float)
    current_set = np.array(current_set).astype(np.float)
    wattage_set = abs(voltage_set) * abs(current_set)
    return voltage_set, current_set, wattage_set

def middle_calculator(array):
    if len(array // 2 == 0):
        middle_value = (array[int(len(array)/2)] + array[int(len(array)/2 - 1)]) / 2
    else:
        middle_value = array[int(len(array)/2)]
    return middle_value

external_stylesheets = ['./stylesheet/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

datasets = find_dataset()

app.layout = html.Div([
    html.Label('Mode'),
    dcc.Dropdown(
        id='mode',
        options=[
            {'label': 'Signal', 'value': 'signal'},
            {'label': 'FFT', 'value': 'FFT'},
        ],
        value='signal'
    ),

    html.Label('Choose the dataset'),
    dcc.Dropdown(
        id='dropdown-dataset',
        options=datasets,
        value=0
    ),

    html.Br(),

    html.Label('Hide or show'),
    dcc.Checklist(
        id='is-shown-checklist',
        options=[
            {'label': 'Voltage', 'value': 'v'},
            {'label': 'Current', 'value': 'c'},
            {'label': 'Wattage', 'value': 'w'}
        ],
        value=['v', 'c']
    ),

    html.Br(),

    html.Label('Sampling rate'),
    dcc.Input(id='sampling-rate', value='152000', type='text'),

    html.Br(),

    html.Label('Time scale'),
    dcc.Slider(
        id='time-scale',
        min=-100,
        max=100,
        marks={i: str(i) for i in range(-100, 110, 10)},
        value=0,
    ),

    html.Label('Time shift'),
    dcc.Slider(
        id='time-shift',
        min=-100,
        max=100,
        marks={i: str(i) for i in range(-100, 110, 10)},
        value=0,
    ),

    # html.Label('Voltage scale'),
    # dcc.Slider(
    #     id='voltage-scale',
    #     min=-100,
    #     max=100,
    #     marks={i: str(i) for i in range(-100, 110, 10)},
    #     value=0,
    # ),

    # html.Label('Current scale'),
    # dcc.Slider(
    #     id='current-scale',
    #     min=-100,
    #     max=100,
    #     marks={i: str(i) for i in range(-100, 110, 10)},
    #     value=0,
    # ),

    # html.Label('Wattage scale'),
    # dcc.Slider(
    #     id='wattage-scale',
    #     min=-100,
    #     max=100,
    #     marks={i: str(i) for i in range(-100, 110, 10)},
    #     value=0,
    # ),
    dcc.Graph(id = 'graph-with-controlls')
], style={'columnCount': 1})

@app.callback(
    Output('graph-with-controlls', 'figure'),
    [
        Input('mode', 'value'),
        Input('dropdown-dataset', 'value'),
        Input('is-shown-checklist','value'),
        Input('sampling-rate', 'value'),
        Input('time-scale', 'value'),
        Input('time-shift', 'value'),
        # Input('voltage-scale', 'value'),
        # Input('current-scale', 'value'),
        # Input('wattage-scale', 'value'),
    ],
    )
def replot_figure(mode, option, shown_list, sampling_rate, time_scale, time_shift): #, voltage_scale, current_scale, wattage_scale):
    global count
    print('replot figure called', count)
    count += 1
    dataset_path = datasets[option]['label']
    voltage_set, current_set, wattage_set = read_data_from_txt(dataset_path)
    sampling_rate = int(sampling_rate)
    time_interval = 1 / float(sampling_rate) 
    time_set = np.arange(0, time_interval * len(voltage_set), time_interval)
    fig = go.Figure()
    fig.data = []
    if mode == 'signal':
        fig.add_trace(go.Scatter(x=time_set, y=voltage_set, name='voltage', visible= 'v' in shown_list, line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=time_set, y=current_set, name='current', visible= 'c' in shown_list, line=dict(color='red'), yaxis='y2'))
        fig.add_trace(go.Scatter(x=time_set, y=wattage_set, name='wattage', visible= 'w' in shown_list, line=dict(color='green'), yaxis='y3'))
        middle_x = middle_calculator(time_set)
    elif mode == 'FFT':
        voltage_set = abs(np.fft.fft(voltage_set))
        current_set = abs(np.fft.fft(current_set))
        wattage_set = abs(np.fft.fft(wattage_set))
        freq_set = np.arange(len(voltage_set))
        middle_x = middle_calculator(freq_set)

        fig.add_trace(go.Scatter(x=freq_set, y=voltage_set, name='voltage', visible= 'v' in shown_list, line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=freq_set, y=current_set, name='current', visible= 'c' in shown_list, line=dict(color='red'), yaxis='y2'))
        fig.add_trace(go.Scatter(x=freq_set, y=wattage_set, name='wattage', visible= 'w' in shown_list, line=dict(color='green'), yaxis='y3'))

    x_interval = middle_x / 100
    x_upper_bound = ( middle_x                                               \
                        + x_interval * (time_scale + 101))                   \
                        + x_interval * time_shift
    x_lower_bound = ( middle_x                                               \
                        - x_interval * (time_scale + 101))                   \
                        + x_interval * time_shift
    fig.update_layout(
    xaxis=dict(
        domain=[0.05, 1],
        range=[x_lower_bound, x_upper_bound]
    ),
    yaxis=dict(
        title="voltage",
        titlefont=dict(
            color="blue"
        ),
        tickfont=dict(
            color="blue"
        ),
        # range=[0.0,1]
    ),
    yaxis2=dict(
        title="current",
        titlefont=dict(
            color="red"
        ),
        tickfont=dict(
            color="red"
        ),
        anchor="free",
        overlaying="y",
        side="left",
        position=0.0
    ),
    yaxis3=dict(
        title="wattage",
        titlefont=dict(
            color="green"
        ),
        tickfont=dict(
            color="green"
        ),
        anchor="x",
        overlaying="y",
        side="right"
    ),
)
    return fig

app.run_server(debug=True)