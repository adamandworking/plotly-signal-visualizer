import dash
import dash_core_components as dcc
import dash_html_components as html
import glob
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash.dependencies import Input, Output, State
import serial
import threading
import time

class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s
    
    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(2048, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)

# Source code from https://github.com/pyserial/pyserial/issues/216#issuecomment-369414522


count = 0
def find_dataset():
    dataset_list = glob.glob('./dataset/*.txt')
    dict_dataset_list = []
    for idx, dataset in enumerate(dataset_list):
        dict_dataset_list.append({'label': dataset, 'value': idx})
    return dict_dataset_list

def read_data_from_stream():
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
voltage_set, current_set, wattage_set = [0],[0],[]
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

    dcc.Graph(id = 'graph-with-controlls'),
    dcc.Interval(
        id='interval-component',
        interval=80, # in milliseconds
        n_intervals=0
    )
], style={'columnCount': 1})

@app.callback(
    Output('graph-with-controlls', 'figure'),
    [
        Input('interval-component', 'n_intervals'),
        Input('mode', 'value'),
        Input('is-shown-checklist','value'),
        Input('sampling-rate', 'value'),
        Input('time-scale', 'value'),
        Input('time-shift', 'value'),
    ],
    )
def replot_figure(n_intervals, mode, shown_list, sampling_rate, time_scale, time_shift):
    global count
    print('replot figure called', count)
    count += 1
    sampling_rate = int(sampling_rate)
    time_interval = 1 / float(sampling_rate) 
    global voltage_set
    global current_set
    global wattage_set
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

def reading_streaming_data(rl):
    global voltage_set
    global current_set
    global wattage_set
    count = 0
    while True:
        voltage_data = np.array(rl.readline().decode().replace('\r', '').replace('\n', '').split(','))
        current_data = np.array(rl.readline().decode().replace('\r', '').replace('\n', '').split(','))
        voltage_data = voltage_data.astype(float)
        current_data = current_data.astype(float)
        voltage_set = voltage_data
        current_set = current_data
        wattage_set = abs(voltage_set) * abs(current_set)
        print('streaming', count)
        count += 1
        # print(voltage_set)
    # return voltage_set, current_set, wattage_set

def run_server(app):
    app.run_server(debug=True)
ser = serial.Serial('COM5', 115200)
rl = ReadLine(ser)
reading_thread = threading.Thread(target= lambda: reading_streaming_data(rl))
reading_thread.setDaemon(True)
reading_thread.start()

# server_thread = threading.Thread(target= lambda: run_server(app))
# server_thread.setDaemon(True)
# server_thread.start()

# reading_streaming_data(rl)
app.run_server(debug=True, use_reloader=False)
