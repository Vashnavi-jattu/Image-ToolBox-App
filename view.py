from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from scipy.optimize import curve_fit, minimize
import math
from scipy.signal import TransferFunction as TF, lsim
import cv2
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
import numpy as np
# import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
from collections import deque

# Define global variable
# global time_data
# global pv_data
# Create your views here.
def control_tuning_imagebased(request):
    return render(request, 'extract_imagedata.html')


@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        # input_value = request.POST.get('input')
        print(request.POST)
        global x_min, x_max, y_min, y_max
        y_min = float(request.POST.get('y_min'))
        y_max = float(request.POST.get('y_max'))
        x_min = float(request.POST.get('x_min'))
        x_max = float(request.POST.get('x_max'))
        op_color = request.POST.get('op-color')
        pv_color = request.POST.get('pv-color')
        # sp_color = request.POST.get('sp-color')
        # sp_color = 'green'
        # Create a dictionary mapping each variable (OP, PV, SP) to its color
        variable_to_color = {
            'OP': op_color,
            'PV': pv_color,
            # 'SP': sp_color
        }

        # Now determine the reverse: which category gets which color
        for var_name, color in variable_to_color.items():
            # if color == 'green':
            #     line_green_name = var_name
            if color == 'blue':
                line_blue_name = var_name
            elif color == 'red':
                line_red_name = var_name
        
        
        image_bytes = image.read()  # Read file as bytes
        # Convert bytes to a NumPy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode the image using OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        # Process the image (save it, run some analysis, etc.)
        # Example: Save image, generate a graph
         # Convert the Plotly figure to an HTML div string
        red_df, red_df_scaled, green_df, green_df_scaled, blue_df, blue_df_scaled = extract_data(img, y_min, y_max, x_min, x_max)
        # global pv_data
        # global op_data
        # pv_data = blue_df_scaled
        # op_data = red_df_scaled
        global pv_data
        global op_data
        if line_blue_name == 'PV':
            pv_data = blue_df_scaled
            op_data = red_df_scaled
        elif line_blue_name == 'OP':
            pv_data = red_df_scaled
            op_data = blue_df_scaled
        # op_data = np.ones_like(red_df_scaled['x'])#*red_df_scaled.max()
        # t0 = get_estimated_timeStepChange(np.array(red_df_scaled['y']), list(red_df_scaled['x']) )
        # op_data[0:int(t0)] = 0
        # time_data, pv_data = get_pv(blue_df_scaled)
        graph_html = plot_extract_data(y_min,y_max,x_min, x_max, red_df, red_df_scaled, line_red_name, green_df, green_df_scaled, blue_df, blue_df_scaled, line_blue_name )
    
        # image_url = image #'path_to_image.jpg'  # Example: Save image and return URL
        # graph_url = 'path_to_graph.jpg'  # Example: Generate graph and return URL
        return JsonResponse({'graph_html': graph_html})
        # return JsonResponse({'image_url': image_url, 'graph_html': graph_html})


def get_hsv_ranges(image, hsv_image, k=3, margin=20):
    """
    Automatically detect dominant colors in an image and define HSV color ranges.

    Args:
        image_path (str): Path to the image.
        k (int): Number of dominant colors to detect (default: 3).
        margin (int): Adjustment margin for lower and upper HSV bounds.

    Returns:
        dict: Dictionary mapping color names to their HSV lower/upper bounds.
    """
    # Read the image
    # image = cv2.imread(image_path)
    # hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)

    
    # Reshape image for clustering
    pixels = hsv_image.reshape(-1, 3)

    # Apply KMeans clustering to find dominant colors
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)  # HSV format

    # Color mapping (you can expand this based on your needs)
    predefined_colors = ["pink", "green", "blue"]  # Adjust as needed
    color_ranges = {}

    for i, color in enumerate(dominant_colors):
        h, s, v = color

        # Define lower and upper bounds with a margin
        lower = np.array([max(h - margin, 50), max(s - margin, 50), max(v - margin, 50)])
        upper = np.array([min(h + margin, 255), min(s + margin, 255), min(v + margin, 255)])

        # Assign names dynamically
        color_name = predefined_colors[i] if i < len(predefined_colors) else f"color_{i+1}"
        color_ranges[f"{color_name}_lower"] = lower
        color_ranges[f"{color_name}_upper"] = upper

    return color_ranges


def extract_data(image, y_min, y_max, x_min, x_max):
    # Load the image
    # image = cv2.imread('/content/drive/MyDrive/Colab Notebooks/LoopTest1.JPG')
    # image = img #cv2.imread(img)
    # y_min = 1440
    # y_max = 1480

    # Convert the image to HSV for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV color ranges for each line color (adjust as necessary)
    # pink_lower, pink_upper = np.array([140, 50, 50]), np.array([170, 255, 255])
    green_lower, green_upper = np.array([50, 100, 50]), np.array([70, 255, 255])
    # lightblue_lower, lightblue_upper = np.array([90, 50, 180]), np.array([110, 255, 255])

    # red_lower = np.array([0, 240, 50])
    # red_upper = np.array([10, 255, 255])
    # blue_lower = np.array([90, 50, 50])
    # blue_upper = np.array([140, 255, 255])

    # Define HSV ranges for red and blue
    lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
    lower_blue, upper_blue = np.array([100, 150, 50]), np.array([140, 255, 255])

    # Create masks for red and blue
    red_mask = cv2.inRange(hsv_image, lower_red1, upper_red1) + cv2.inRange(hsv_image, lower_red2, upper_red2)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)


    # # Auto Detect HSV
    # hsv_ranges = get_hsv_ranges(image, hsv_image)

    # # Extract named variables dynamically
    # for name, value in hsv_ranges.items():
    #     globals()[name] = value  # Creates variables dynamically

    # # Print results
    # print(f"pink_lower = {pink_lower}, pink_upper = {pink_upper}")
    # print(f"green_lower = {green_lower}, green_upper = {green_upper}")
    # print(f"blue_lower = {blue_lower}, blue_upper = {blue_upper}")



    # Create color masks
    # pink_mask = cv2.inRange(hsv_image, pink_lower, pink_upper)
    # red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    # blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
    # lightblue_mask = cv2.inRange(hsv_image, lightblue_lower, lightblue_upper)
    
    # Find pixel coordinates for each color
    # pink_coords = np.column_stack(np.where(pink_mask > 0))
    red_coords = np.column_stack(np.where(red_mask > 0))
    green_coords = np.column_stack(np.where(green_mask > 0))
    blue_coords = np.column_stack(np.where(blue_mask > 0))
    # lightblue_coords = np.column_stack(np.where(lightblue_mask > 0))
    
    # Invert the y-coordinates
    # pink_coords[:, 0] = pink_mask.shape[0] - pink_coords[:, 0]
    red_coords[:, 0] = red_mask.shape[0] - red_coords[:, 0]
    green_coords[:, 0] = green_mask.shape[0] - green_coords[:, 0]
    blue_coords[:, 0] = blue_mask.shape[0] - blue_coords[:, 0]
    # lightblue_coords[:, 0] = lightblue_mask.shape[0] - lightblue_coords[:, 0]

    # pink_df = pd.DataFrame(pink_coords, columns=['y', 'x'])
    red_df = pd.DataFrame(red_coords, columns=['y', 'x'])
    green_df = pd.DataFrame(green_coords, columns=['y', 'x'])
    blue_df  = pd.DataFrame(blue_coords, columns=['y', 'x'])
    
    red_df_scaled = pd.DataFrame()
    green_df_scaled = pd.DataFrame()
    blue_df_scaled = pd.DataFrame()

    # Ensure data is sorted by x values
    red_df = red_df.sort_values(by="x").reset_index(drop=True)
    blue_df = blue_df.sort_values(by="x").reset_index(drop=True)

    # Identify the bigger and smaller DataFrame
    if len(red_df) > len(blue_df):
        big_df, small_df = red_df, blue_df
    else:
        big_df, small_df = blue_df, red_df

    # Find missing x values in the smaller DataFrame
    missing_x = big_df[~big_df["x"].isin(small_df["x"])]

    # Merge missing rows into the smaller DataFrame
    small_df = pd.concat([small_df, missing_x]).sort_values(by="x").reset_index(drop=True)

    # Restore original variable names
    if len(red_df) > len(blue_df):
        blue_df = small_df  # Smaller DataFrame was blue, now completed
    else:
        red_df = small_df  # Smaller DataFrame was red, now completed



    # lightblue_df = pd.DataFrame(lightblue_coords, columns=['y', 'x'])
    
    # if not red_df.empty:
    #     red_df_scaled = scale_values(red_df, y_min, y_max, x_min, x_max)
    #     red_df_processed = interpolate_df(red_df_scaled)
    if not red_df.empty:
        red_df_processed = interpolate_df(red_df,x_min,x_max)
        red_df_scaled = scale_values(red_df_processed, y_min, y_max, x_min, x_max) 
    # if not lightblue_df.empty:
    #     lightblue_df_processed = interpolate_df(lightblue_df)
    #     lightblue_df_scaled = scale_values(lightblue_df_processed,  y_min, y_max)
    # if not pink_df.empty:
    #     pink_df_processed = interpolate_df(pink_df)
    #     pink_df_scaled = scale_values(pink_df_processed,  y_min, y_max)
    if not green_df.empty:
        green_df_processed = interpolate_df(green_df,x_min,x_max)
        green_df_scaled = scale_values(green_df_processed, y_min, y_max, x_min, x_max)
    if not blue_df.empty:
        blue_df_processed = interpolate_df(blue_df,x_min,x_max)
        blue_df_scaled = scale_values(blue_df_processed, y_min, y_max, x_min, x_max)

    # prompt: remove outliers and interpolate green, blue and pink_df and create a new dataframe for that
    # Scale the y-values
    # pink_df_scaled = scale_values(pink_df_processed, 0, len(pink_mask), y_min, y_max)
    # green_df_scaled = scale_values(green_df_processed, 0, len(pink_mask), y_min, y_max)
    # blue_df_scaled = scale_values(blue_df_processed, 0, len(pink_mask), y_min, y_max)


    return red_df, red_df_scaled, green_df, green_df_scaled, blue_df, blue_df_scaled


def plot_extract_data(y_min,y_max,x_min, x_max, red_df, red_df_scaled, line_red_name, green_df, green_df_scaled, blue_df, blue_df_scaled, line_blue_name ):
    
    fig = go.Figure()
    if not red_df.empty:
        fig.add_trace(go.Scatter(x=red_df_scaled.x, y=red_df_scaled.y,
                            mode='lines',
                            line=dict(color='red'), # Set line color to limegreen
                           name=line_red_name))
    # if not green_df.empty:
    #     fig.add_trace(go.Scatter(x=green_df_scaled.x, y=green_df_scaled.y,
    #                         mode='lines',
    #                         line=dict(color='green'), # Set line color to limegreen
    #                        name=line_green_name))
    # if not pink_df.empty:
    #     fig.add_trace(go.Scatter(x=pink_df_processed.index, y=pink_df_scaled.y,
    #                         mode='lines',
    #                         line=dict(color='hotpink'), # Set line color to hotpink
    #                         name=line_red_name))
    if not blue_df.empty:
        fig.add_trace(go.Scatter(x=blue_df_scaled.x, y=blue_df_scaled.y,
                            mode='lines',
                            line=dict(color='blue'), # Set line color to deepskyblue
                            name=line_blue_name))
    # if not lightblue_df.empty:
    #     fig.add_trace(go.Scatter(x=lightblue_df_processed.index, y=lightblue_df_scaled.y,
    #                         mode='lines',
    #                         line=dict(color='deepskyblue'), # Set line color to deepskyblue
    #                         name=line_blue_name))

    fig.update_layout(#title='Extracted Data',
                    # xaxis_title='Time',
                    # yaxis_title='Units',
                    template="plotly_white",
                    autosize=True,
                    height=450,  # Set initial height
                    width=1000,  # Set initial width
                    )
    fig.update_yaxes(range=[y_min, y_max])
    fig.update_xaxes(range=[x_min, x_max])
    # fig.show()
     # Convert the Plotly figure to an HTML div string
    graph_html = fig.to_html(full_html=False)
    
    return graph_html


def remove_outliers_iqr(df, column='y'):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered


def interpolate_df(df,new_xmin,new_xmax):
    df = remove_outliers_iqr(df, 'y')
    x = df['x'].values
    y = df['y'].values

    if len(set(x)) != len(x):
        x = [xi + np.random.uniform(-0.1, 0.1) for xi in x]

    
    x_new = np.linspace(min(x), max(x), num=int(10*(new_xmax-new_xmin))+1)
    
    f = interp1d(x, y, kind='linear')
    # x_new = np.linspace(x_min, x_max, num=251)
    y_new = f(x_new)
    return pd.DataFrame({'x': x_new, 'y': y_new})


def scale_values(df, new_ymin, new_ymax, new_xmin,new_xmax):
    """Scales values in a DataFrame column from one range to another."""

    old_ymin = df['y'].min()
    old_ymax = df['y'].max()
    df['y'] = new_ymin + ((df['y'] - old_ymin) / (old_ymax - old_ymin)) * (new_ymax - new_ymin)


    old_xmax = df['x'].max()
    old_xmin = df['x'].min()

    df['x'] = new_xmin + ((df['x'] - old_xmin) / (old_xmax - old_xmin)) * (new_xmax - new_xmin)
    return df


# Calculate the baseline as the mean of the initial few samples
def get_estimated_timeStepChange(signal, time):
  baseline_window_size = 5
  baseline = np.mean(signal[:baseline_window_size])  # Adjust as needed for your signal

  # Identify when the signal significantly deviates from the baseline
  threshold = baseline + 0.02 * (np.max(signal) - baseline)  # Adjust threshold as needed
  step_start_index = np.where(signal > threshold)[0][0]  # First point above the threshold
  estimated_delay = time[step_start_index] - time[0]
#   if estimated_delay < baseline_window_size:
#       estimated_delay = 0
  print(f"Estimated Time: {estimated_delay} time units")
  return estimated_delay


# @csrf_exempt
# def first_order_plus_dead_time(t, Kp, tau, theta):
#     """ First-Order Plus Dead-Time (FOPTD) model response to a step input """
#     y = Kp * (1 - np.exp(-(t - theta) / tau))  # Step response formula
#     y[t < theta] = 0  # Before delay time, output is zero
#     return y

# def identify_foptd(request):
#     """
#     Identify the First-Order Plus Time Delay (FOPTD) model parameters from step response data.
    
#     Parameters:
#     t : array-like
#         Time vector
#     y : array-like
#         Process response data
    
#     Returns:
#     Kp : float
#         Process gain
#     tau : float
#         Time constant
#     theta : float
#         Time delay
#     """

     
#     t = list(pv_data['x']), 
#     t=t[0]
#     y = np.array(pv_data['y'])
#     OP = np.array(op_data['y'])
#     # print(t, process_data)


#     t0 = get_estimated_timeStepChange(OP, t)
#     td = get_estimated_timeStepChange(y, t)

#     # Estimate Kp (steady-state gain)
#     Kp = (y[-1] - y[0]) / (1.0)  # Assuming input step of 1.0

#     # Estimate delay (theta) as the time when output reaches 10% of final value
#     y10 = y[0] + 0.1 * (y[-1] - y[0])
#     theta_idx = np.where(y >= y10)[0][0]  # Find index where response reaches 10%
#     theta = t[theta_idx]

#     # Estimate tau using 63% rise time (time when response reaches 63% of final value)
#     y63 = y[0] + 0.63 * (y[-1] - y[0])
#     tau_idx = np.where(y >= y63)[0][0]  # Find index where response reaches 63%
#     tau = t[tau_idx] - theta

#     # Curve fitting to refine parameters
#     initial_guess = [Kp, tau, theta]
#     params, _ = curve_fit(first_order_plus_dead_time, t, y, p0=initial_guess, bounds=(0, np.inf))
#     global Kp_fit, tau_fit, theta_fit
#     Kp_fit, tau_fit, theta_fit = params
#     # return Kp_fit, tau_fit, theta_fit
#     ymodel = first_order_plus_dead_time(t, Kp_fit, tau_fit, theta_fit)

    

#     fig = go.Figure()

#     fig.add_trace(go.Scatter(x=t, y=ymodel,
#                             mode='lines',
#                             name='Fitted Model'))

#     fig.add_trace(go.Scatter(x=t, y=y,
#                             mode='lines',
#                             name='Process Data'))

#     # fig.add_trace(go.Scatter(x=[t0], y=[y0],
#     #                         mode='markers',
#     #                         marker=dict(color='red', size=10),
#     #                         name='t0, y0'))

#     # fig.add_trace(go.Scatter(x=[td], y=[y0],
#     #                         mode='markers',
#     #                         marker=dict(color='green', size=10),
#     #                         name='td, y0'))

#     # fig.add_trace(go.Scatter(x=[t1], y=[y_tau],
#     #                         mode='markers',
#     #                         marker=dict(color='blue', size=10),
#     #                         name='t1, y_tau'))

#     # fig.add_shape(go.layout.Shape(
#     #     type="line",
#     #     x0=min(t),
#     #     y0=y_inf,
#     #     x1=max(t),
#     #     y1=y_inf,
#     #     line=dict(color="purple", width=2, dash="dash"),
#     # ))

#     # fig.add_shape(go.layout.Shape(
#     #     type="line",
#     #     x0=min(t),
#     #     y0=y0,
#     #     x1=max(t),
#     #     y1=y0,
#     #     line=dict(color="orange", width=2, dash="dash"),
#     # ))

#     # fig.add_shape(go.layout.Shape(
#     #     type="line",
#     #     x0=min(t),
#     #     y0=y_tau,
#     #     x1=max(t),
#     #     y1=y_tau,
#     #     line=dict(color="gray", width=2, dash="dash"),
#     # ))


#     fig.update_layout(title='FOPTD Model Identification',
#                     xaxis_title='Time',
#                     yaxis_title='Process Output',
#                     template="plotly_white",
#                     autosize=True,
#                     annotations=[
#                         # dict(x=t0, y=y0, text=f"t0", showarrow=True, arrowhead=7),
#                         # dict(x=td, y=y0, text=f"td = {td:.2f}", showarrow=True, arrowhead=7),
#                         # dict(x=t1, y=y_tau, text=f"t1 = {t1:.2f}, y_tau = {y_tau:.2f}", showarrow=True, arrowhead=7),
#                         # dict(x=t1, y=y_inf, text=f"y_inf = {y_inf:.2f}", showarrow=True, arrowhead=7),
#                         dict(x=0.95, y=0.4, xref="paper", yref="paper",
#                             text=f"theta_fit : {theta_fit:.2f}<br>tau_fit : {tau_fit:.2f}<br>Kp_fit: {Kp_fit:.2f}",
#                             showarrow=False)
#                     ]
#                     )

#     # fig.show()

#     # Convert the Plotly figure to an HTML div string
#     modeling_graph_html = fig.to_html(full_html=False)

#     return JsonResponse({'modeling_graph_html': modeling_graph_html})



def identify_foptd(request):
    """
    Identify the first-order plus time delay (FOPTD) model parameters.

    Parameters:
        t (array-like): Time data.
        process_data (array-like): Process response data.
        initial_guess (tuple): Initial guess for (Kp, tau, theta).

    Returns:
        tuple: Identified parameters (Kp, tau, theta).
    """
    global modeling_type
    modeling_type = 'First-Order Plus Dead Time (FOPDT)'
    t = list(pv_data['x']), 
    t=t[0]
    process_data = np.array(pv_data['y'])
    OP = np.array(op_data['y'])
    # print(t, process_data)


    t0 = get_estimated_timeStepChange(OP, t)
    td = get_estimated_timeStepChange(process_data, t)
    
    start_index = int(t0 * 10)  # Convert t0 (seconds) to index
    t = t[start_index:]  # Slice from t0 onward
    t = [x - t[0] for x in t]  # Reset so first value is 0
    process_data = process_data[int(t0*10):]
    # t = t[int(t0):]
    # process_data = process_data[int(t0):]
    # estimated_delay = td-t0
    # # Parameters
    # u_inf = OP[-1]
    # u_0 = OP[0]
    # # y0 = np.mean(process_data[:int(td)])
    # y0 = process_data[int(td)]
    # y_inf = np.mean(process_data[-10:])
    # y_tau = (y_inf - y0)*0.632 + y0
    # x_tau = t[process_data[y_tau]]
    # print(u_0, u_inf, y0, y_inf, y_tau)
    # # t0 = time[0]
    # # td = t[int(estimated_delay)]
    # # Find the index where the signal is closest to y_tau
    # # idx = (process_data - y_tau).abs().idxmin()
    # # # Get the corresponding x value
    # # t1 = t[idx]

    # # time_delay = td - t0
    # print(f"Time delay: {estimated_delay}")
    # # time_constant = t1 - td
    # # print(f"Time constant(tau): {time_constant}")
    # k_gain = (y_inf - y0)/(u_inf - u_0)
    # print(f"K-gain: {k_gain}")

    # global Kp_fit, tau_fit, theta_fit
    # Kp_fit, tau_fit, theta_fit = k_gain, x_tau, estimated_delay

    # Sample data
    # t = np.linspace(0, 10, 100)
    # process_data = np.sin(t)
    # print(t, process_data)

    def first_order_tf_with_delay(t, Kp, tau, theta):
        A = OP[-1] - OP[0] #1  # Magnitude of step change in input
        ymodel = np.zeros_like(t)  # Initialize the response array

        for i, time in enumerate(t):
            if time >= theta:
                ymodel[i] = A * Kp * (1 - np.exp(-(time - theta) / tau))

        return ymodel
    
    
    # Initial guesses for Kp, tau, theta
    initial_guess = [1.0, 1.0, 1.0]
    # Fit the model
    params, params_covariance = curve_fit(first_order_tf_with_delay, t, process_data, p0=initial_guess)
    global Kp_fit, tau_fit, theta_fit
    
    Kp_fit, tau_fit, theta_fit = params
    # theta_fit = theta_fit - t0

    global modeling_params
    modeling_params = params
    global modeling_params_names
    modeling_params_names = ['Kp', 'Tau', 'Theta']

    ymodel = first_order_tf_with_delay(t, Kp_fit, tau_fit, theta_fit)

    
    
    # # 
    # def detect_op_change_time(t, OP, threshold=0.2):
    #     """Find the first significant change in OP signal."""
    #     for i in range(1, len(OP)):
    #         if abs(OP[i] - OP[0]) > threshold:  # Detects when OP changes significantly
    #             return t[i]  # Return the time of first significant change
    #     return t[0]  # Default to the first time point if no change detected

    # # Example: Detect change
    # theta_initial = detect_op_change_time(t, OP)

    # # Initial guesses for Kp, tau, theta
    # initial_guess = [1.0, 1.0, theta_initial]

    # def first_order_tf_with_delay_op(t, Kp, tau, theta, OP, t_OP):
    #     """First-order system with dead time and OP signal."""
    #     ymodel = np.zeros_like(t)

    #     for i, time in enumerate(t):
    #         if time >= theta:
    #             # Find the corresponding OP value at t-theta
    #             op_index = np.searchsorted(t_OP, time - theta)  # Find nearest index in OP
    #             op_value = OP[op_index] if op_index < len(OP) else OP[-1]  # Avoid out-of-bounds
    #             ymodel[i] = Kp * op_value * (1 - np.exp(-(time - theta) / tau))  
    #         else:
    #             ymodel[i] = 0  # No response before delay

    #     return ymodel


    # # # Example: Fit with OP signal
    # # # Curve fitting with OP
    # params, _ = curve_fit(lambda t, Kp, tau, theta: first_order_tf_with_delay_op(t, Kp, tau, theta, OP, t),
    #                   t, process_data, p0=initial_guess)
    # global Kp_fit, tau_fit, theta_fit
    # Kp_fit, tau_fit, theta_fit = params
    # ymodel = first_order_tf_with_delay_op(t, Kp_fit, tau_fit, theta_fit, OP, t)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=ymodel,
                            mode='lines',
                            name='Fitted Model'))

    fig.add_trace(go.Scatter(x=t, y=process_data,  ### sample rate = 10
                            mode='lines',
                            name='Process Data'))

    # fig.add_trace(go.Scatter(x=[t0], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='red', size=10),
    #                         name='t0, y0'))

    # fig.add_trace(go.Scatter(x=[td], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='green', size=10),
    #                         name='td, y0'))

    # fig.add_trace(go.Scatter(x=[t1], y=[y_tau],
    #                         mode='markers',
    #                         marker=dict(color='blue', size=10),
    #                         name='t1, y_tau'))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_inf,
    #     x1=max(t),
    #     y1=y_inf,
    #     line=dict(color="purple", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y0,
    #     x1=max(t),
    #     y1=y0,
    #     line=dict(color="orange", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_tau,
    #     x1=max(t),
    #     y1=y_tau,
    #     line=dict(color="gray", width=2, dash="dash"),
    # ))


    fig.update_layout(title='FOPTD Model Identification',
                    xaxis_title='Time',
                    yaxis_title='Process Output',
                    template="plotly_white",
                    autosize=True,
                    annotations=[
                        # dict(x=t0, y=y0, text=f"t0", showarrow=True, arrowhead=7),
                        # dict(x=td, y=y0, text=f"td = {td:.2f}", showarrow=True, arrowhead=7),
                        # dict(x=t1, y=y_tau, text=f"t1 = {t1:.2f}, y_tau = {y_tau:.2f}", showarrow=True, arrowhead=7),
                        # dict(x=t1, y=y_inf, text=f"y_inf = {y_inf:.2f}", showarrow=True, arrowhead=7),
                        dict(x=0.95, y=0.4, xref="paper", yref="paper",
                            text=f"theta_fit : {theta_fit:.2f}<br>tau_fit : {tau_fit:.2f}<br>Kp_fit: {Kp_fit:.2f}",
                            showarrow=False)
                    ]
                    )

    # fig.show()
    global modeling_graph
    modeling_graph = fig
    # Convert the Plotly figure to an HTML div string
    modeling_graph_html = fig.to_html(full_html=False)

    return JsonResponse({'modeling_graph_html': modeling_graph_html})

    # return JsonResponse({'Kp_fit': Kp_fit, 'tau_fit': tau_fit,'theta_fit':theta_fit})


def identify_soptd(request):
    """
    Identify the second-order plus time delay (SOPTD) model parameters.

    Parameters:
        t (array-like): Time data.
        process_data (array-like): Process response data.
        initial_guess (tuple): Initial guess for (Kp, tau1, tau2, theta).

    Returns:
        tuple: Identified parameters (Kp, tau1, tau2, theta).
    """
    global modeling_type
    modeling_type = 'Second-Order Plus Dead Time (SOPDT)'
    t = list(pv_data['x']), 
    t=t[0]
    process_data = np.array(pv_data['y'])
    OP = np.array(op_data['y'])
    t0 = get_estimated_timeStepChange(OP, t)
    td = get_estimated_timeStepChange(process_data, t)
    
    start_index = int(t0 * 10)  # Convert t0 (seconds) to index
    t = t[start_index:]  # Slice from t0 onward
    t = [x - t[0] for x in t]  # Reset so first value is 0
    process_data = process_data[int(t0*10):]

    def second_order_tf_with_delay(t, Kp, tau, zeta, Θ):
        A = 1 #OP[-1] - OP[0] #1
        response = np.zeros_like(t)
        for i, time in enumerate(t):
            if time >= Θ:
                timeshift = time - Θ
                if zeta > 1:  # overdamped process
                    AA1 = math.cosh(np.sqrt(zeta ** 2 - 1) * (timeshift / tau))
                    AA0 = zeta / np.sqrt(zeta ** 2 - 1)
                    AA2 = math.sinh(np.sqrt(zeta ** 2 - 1) * (timeshift / tau))
                    response[i] = A * Kp * (1 - np.exp(-(timeshift * zeta) / tau) * (AA1 + AA0 * AA2))
                elif zeta == 1:  # critically damped process
                    response[i] = A * Kp * (1 - (1 + (timeshift / tau)) * np.exp(-timeshift / tau))
                else:  # underdamped process
                    BB0 = 1 / np.sqrt(1 - zeta ** 2)
                    omega = np.sqrt(1 - zeta ** 2) / tau
                    phi = math.atan(np.sqrt(1 - zeta ** 2) / zeta)
                    response[i] = A * Kp * (
                                1 - BB0 * np.exp(-(timeshift * zeta) / tau) * math.sin(omega * timeshift + phi))
        return response

    # Initial guesses for Kp, tau1, tau2, theta
    initial_guess = [1.0, 1.0, 0.5, 1.0]
    # Fit the model
    
    params, params_covariance = curve_fit(second_order_tf_with_delay, t, process_data, p0=initial_guess)
    Kp_est, tau_est, zeta_est, Θ_est = params

    global modeling_params
    modeling_params = params
    global modeling_params_names
    modeling_params_names = ['Kp', 'Tau', 'Zeta', 'Theta']

    ymodel = second_order_tf_with_delay(t, Kp_est, tau_est, zeta_est, Θ_est)

    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=ymodel,
                            mode='lines',
                            name='Fitted Model'))

    fig.add_trace(go.Scatter(x=t, y=process_data,
                            mode='lines',
                            name='Process Data'))

    # fig.add_trace(go.Scatter(x=[t0], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='red', size=10),
    #                         name='t0, y0'))

    # fig.add_trace(go.Scatter(x=[td], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='green', size=10),
    #                         name='td, y0'))

    # fig.add_trace(go.Scatter(x=[t1], y=[y_tau],
    #                         mode='markers',
    #                         marker=dict(color='blue', size=10),
    #                         name='t1, y_tau'))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_inf,
    #     x1=max(t),
    #     y1=y_inf,
    #     line=dict(color="purple", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y0,
    #     x1=max(t),
    #     y1=y0,
    #     line=dict(color="orange", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_tau,
    #     x1=max(t),
    #     y1=y_tau,
    #     line=dict(color="gray", width=2, dash="dash"),
    # ))


    fig.update_layout(title='SOPTD Model Identification',
                    xaxis_title='Time',
                    yaxis_title='Process Output',
                    template="plotly_white",
                    autosize=True,
                    annotations=[
                        # dict(x=t0, y=y0, text=f"t0", showarrow=True, arrowhead=7),
                        # dict(x=td, y=y0, text=f"td = {td:.2f}", showarrow=True, arrowhead=7),
                        # dict(x=t1, y=y_tau, text=f"t1 = {t1:.2f}, y_tau = {y_tau:.2f}", showarrow=True, arrowhead=7),
                        # dict(x=t1, y=y_inf, text=f"y_inf = {y_inf:.2f}", showarrow=True, arrowhead=7),
                        dict(x=0.95, y=0.4, xref="paper", yref="paper",
                            text=f"Θ_est : {Θ_est:.2f}<br>tau_est : {tau_est:.2f}<br>Kp_est: {Kp_est:.2f}<br>zeta_est: {zeta_est:.2f}",
                            showarrow=False)
                    ]
                    )

    # fig.show() Kp_est, tau_est, zeta_est, Θ_est
    global modeling_graph
    modeling_graph = fig
    # Convert the Plotly figure to an HTML div string
    modeling_graph_html = fig.to_html(full_html=False)

    return JsonResponse({'modeling_graph_html': modeling_graph_html})


def identify_integrator_delay(request):
    # define response of integrator with time delay to a step change in input
    global modeling_type
    modeling_type = 'Integrator Plus Dead Time (IPDT)'

    t = list(pv_data['x']), 
    t=t[0]
    process_data = np.array(pv_data['y'])
    OP = np.array(op_data['y'])
    t0 = get_estimated_timeStepChange(OP, t)
    td = get_estimated_timeStepChange(process_data, t)
    
    start_index = int(t0 * 10)  # Convert t0 (seconds) to index
    t = t[start_index:]  # Slice from t0 onward
    t = [x - t[0] for x in t]  # Reset so first value is 0
    process_data = process_data[int(t0*10):]

    def integrator_with_delay(t, Kp, theta):
        A = 1 #OP[-1] - OP[0] #1  # magnitude of step change in input
        ymodel = np.zeros_like(t)
        for i, time in enumerate(t):
            if time >= theta:
                ymodel[i] = A * Kp * (time - theta)
        return ymodel

    # Identify the FOPTD model from the process data
    initial_guess = [1, 1]
    params, params_covariance = curve_fit(integrator_with_delay, t, process_data, p0=initial_guess)
    Kp_est, theta_est = params

    global modeling_params
    modeling_params = params
    global modeling_params_names
    modeling_params_names = ['Kp', 'Theta']

    # generate response of the process model
    ymodel = integrator_with_delay(t, Kp_est, theta_est)

    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=ymodel,
                            mode='lines',
                            name='Fitted Model'))

    fig.add_trace(go.Scatter(x=t, y=process_data,
                            mode='lines',
                            name='Process Data'))

    # fig.add_trace(go.Scatter(x=[t0], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='red', size=10),
    #                         name='t0, y0'))

    # fig.add_trace(go.Scatter(x=[td], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='green', size=10),
    #                         name='td, y0'))

    # fig.add_trace(go.Scatter(x=[t1], y=[y_tau],
    #                         mode='markers',
    #                         marker=dict(color='blue', size=10),
    #                         name='t1, y_tau'))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_inf,
    #     x1=max(t),
    #     y1=y_inf,
    #     line=dict(color="purple", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y0,
    #     x1=max(t),
    #     y1=y0,
    #     line=dict(color="orange", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_tau,
    #     x1=max(t),
    #     y1=y_tau,
    #     line=dict(color="gray", width=2, dash="dash"),
    # ))


    fig.update_layout(title='Integrator Delay Model Identification',
                    xaxis_title='Time',
                    yaxis_title='Process Output',
                    template="plotly_white",
                    autosize=True,
                    annotations=[
                        # dict(x=t0, y=y0, text=f"t0", showarrow=True, arrowhead=7),
                        # dict(x=td, y=y0, text=f"td = {td:.2f}", showarrow=True, arrowhead=7),
                        # dict(x=t1, y=y_tau, text=f"t1 = {t1:.2f}, y_tau = {y_tau:.2f}", showarrow=True, arrowhead=7),
                        # dict(x=t1, y=y_inf, text=f"y_inf = {y_inf:.2f}", showarrow=True, arrowhead=7),
                        dict(x=0.95, y=0.4, xref="paper", yref="paper",
                            text=f"theta_est : {theta_est:.2f}<br>Kp_est: {Kp_est:.2f}",
                            showarrow=False)
                    ]
                    )

    # fig.show() Kp_est, theta_est
    global modeling_graph
    modeling_graph = fig
    # Convert the Plotly figure to an HTML div string
    modeling_graph_html = fig.to_html(full_html=False)

    return JsonResponse({'modeling_graph_html': modeling_graph_html})


def identify_inverse_response_tf(request):
    global modeling_type
    modeling_type = 'Inverse Response'
    # define response of the model to a step change in input
    t = list(pv_data['x']), 
    t=t[0]
    process_data = np.array(pv_data['y'])
    OP = np.array(op_data['y'])
    t0 = get_estimated_timeStepChange(OP, t)
    td = get_estimated_timeStepChange(process_data, t)
    
    start_index = int(t0 * 10)  # Convert t0 (seconds) to index
    t = t[start_index:]  # Slice from t0 onward
    t = [x - t[0] for x in t]  # Reset so first value is 0
    process_data = process_data[int(t0*10):]

    def inverse_response_model_with_delay(t, Kp, taun, tau1, tau2, theta):
        A = 1 # OP[-1] - OP[0] #1  # magnitude of step change in input
        ymodel = np.zeros_like(t)
        for i, time in enumerate(t):
            if time >= theta:
                AA1 = ((taun - tau1) / (tau1 - tau2))
                AA2 = ((taun - tau2) / (tau2 - tau1))
                BB1 = np.exp(-(time - theta) / tau1)
                BB2 = np.exp(-(time - theta) / tau2)
                ymodel[i] = A * Kp * (1 + AA1 * BB1 + AA2 * BB2)
        return ymodel

    # Identify the FOPTD model from the process data
    initial_guess = [1, -1, 1, 2, 1]
    params, params_covariance = curve_fit(inverse_response_model_with_delay, t, process_data, p0=initial_guess)
    Kp_est, taun_est, tau1_est, tau2_est, theta_est = params

    global modeling_params
    modeling_params = params
    global modeling_params_names
    modeling_params_names = ['Kp', 'Taun', 'Tau1', 'Tau2', 'Theta']

    # generate response of the process model
    ymodel = inverse_response_model_with_delay(t,Kp_est,taun_est,tau1_est,tau2_est, theta_est)
    

    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=ymodel,
                            mode='lines',
                            name='Fitted Model'))

    fig.add_trace(go.Scatter(x=t, y=process_data,
                            mode='lines',
                            name='Process Data'))

    # fig.add_trace(go.Scatter(x=[t0], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='red', size=10),
    #                         name='t0, y0'))

    # fig.add_trace(go.Scatter(x=[td], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='green', size=10),
    #                         name='td, y0'))

    # fig.add_trace(go.Scatter(x=[t1], y=[y_tau],
    #                         mode='markers',
    #                         marker=dict(color='blue', size=10),
    #                         name='t1, y_tau'))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_inf,
    #     x1=max(t),
    #     y1=y_inf,
    #     line=dict(color="purple", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y0,
    #     x1=max(t),
    #     y1=y0,
    #     line=dict(color="orange", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_tau,
    #     x1=max(t),
    #     y1=y_tau,
    #     line=dict(color="gray", width=2, dash="dash"),
    # ))


    fig.update_layout(title='Inverse Response Model Identification',
                    xaxis_title='Time',
                    yaxis_title='Process Output',
                    template="plotly_white",
                    autosize=True,
                    annotations=[
                        # dict(x=t0, y=y0, text=f"t0", showarrow=True, arrowhead=7),
                        # dict(x=td, y=y0, text=f"td = {td:.2f}", showarrow=True, arrowhead=7),
                        # dict(x=t1, y=y_tau, text=f"t1 = {t1:.2f}, y_tau = {y_tau:.2f}", showarrow=True, arrowhead=7),
                        # dict(x=t1, y=y_inf, text=f"y_inf = {y_inf:.2f}", showarrow=True, arrowhead=7),
                        dict(x=0.95, y=0.4, xref="paper", yref="paper",
                            text=f"theta_est : {theta_est:.2f}<br>Kp_est: {Kp_est:.2f}<br>taun_est: {taun_est:.2f}<br>tau1_est: {tau1_est:.2f}<br>tau2_est: {tau2_est:.2f}",
                            showarrow=False)
                    ]
                    )

    # fig.show()  taun_est, tau1_est, tau2_est,  
    global modeling_graph
    modeling_graph = fig
    # Convert the Plotly figure to an HTML div string
    modeling_graph_html = fig.to_html(full_html=False)

    return JsonResponse({'modeling_graph_html': modeling_graph_html})


def design_imc_pid(controller_type, Kp, Tau, theta, lambda1):
    """
    Design IMC-based PID controllers for FOPTD processes.
    """
    if controller_type == "PIDF":
        Kc = (Tau + 0.5 * theta) / (Kp * (theta + lambda1))
        TauI = Tau + 0.5 * theta
        Taud = (Tau * theta) / (2 * Tau + theta)
        TauF = (lambda1 * theta) / (2 * (lambda1 + theta))
        params = Kc, TauI, Taud, TauF
    elif controller_type == "PID":
        Kc = (Tau + 0.5 * theta) / (Kp * (0.5 * theta + lambda1))
        TauI = Tau + 0.5 * theta
        Taud = (Tau * theta) / (2 * Tau + theta)
        params = Kc, TauI, Taud
    elif controller_type == "PI":
        Kc = (Tau + 0.5 * theta) / (Kp * lambda1)
        TauI = Tau + 0.5 * theta
        params = Kc, TauI
    else:
        raise ValueError("Invalid controller type. Use 'PIDF', 'PID', or 'PI'.")
    return params


# @csrf_exempt
# def simulate_imc_pid_fotpd(request):
#     """
#     Simulate the closed-loop response for the IMC-based PID controller using identified parameters.
#     """
#     controller_type = request.POST.get('controller_type')
#     t = list(pv_data['x']), 
#     t=t[0]
#     process_data = np.array(pv_data['y'])
#     OP = np.array(op_data['y'])
#     # if not hasattr(self, "sp") or self.sp is None:
#     #     raise ValueError("Setpoint (sp) is not defined. Perform data extraction first.")

#     # if not hasattr(self, "pv") or self.pv is None:
#     #     raise ValueError("Process variable (pv) is not defined. Perform data extraction first.")

#     # # Use the identified process parameters
#     # if not hasattr(self, "kp") or not hasattr(self, "tau") or not hasattr(self, "theta"):
#     #     raise ValueError("Model parameters (Kp, Tau, Theta) are not defined. Perform identification first.")

#     Kp, Tau, theta = Kp_fit, tau_fit, theta_fit
#     if controller_type == "PIDF":
#         lambda1 = 0.3 * theta
#     elif controller_type == "PID":
#         lambda1 = 0.95 * theta
#     elif controller_type == "PI":
#         lambda1 = 1.9 * theta
    
#     # Create the process model using Pade approximation for the delay
#     # model1: First-order process dynamics
#     model1 = TF([Kp], [Tau, 1])

#     # model2: Pade approximation for the time delay
#     model2 = TF([-0.5 * theta, 1], [0.5 * theta, 1])

#     # Combine model1 and model2 to form the complete process model
#     process_model = TF(np.polymul(model1.num, model2.num), np.polymul(model1.den, model2.den))

#     # Retrieve time data and setpoint from pv and sp
#     # time_data = np.array(self.pv.x)
#     # time_data = np.linspace(0, 50, 500)
#     ysp = np.ones_like(t)#op
#     ysp[0] = 0

#     # Design the controller
#     params = design_imc_pid(controller_type, Kp, Tau, theta, lambda1)

#     # Define the controller transfer function
#     if controller_type == "PIDF":
#         Kc, TauI, Taud, TauF = params
#         Gc1 = TF([Kc * TauI * Taud, Kc * TauI, Kc], [TauI, 0])
#         Gc2 = TF([0, 1], [TauF, 1])
#         Gc = TF(np.polymul(Gc1.num, Gc2.num), np.polymul(Gc1.den, Gc2.den))
#     elif controller_type == "PID":
#         Kc, TauI, Taud = params
#         Gc = TF([Kc * TauI * Taud, Kc * TauI, Kc], [TauI, 0])
#     else:
#         Kc, TauI = params
#         Gc = TF([Kc * TauI, Kc], [TauI, 0])
    
#     closed_loop_numerator = np.polymul(process_model.num, Gc.num)
#     closed_loop_denominator = np.polyadd(
#         np.polymul(process_model.den, Gc.den), np.polymul(process_model.num, Gc.num)
#     )
#     closed_loop_model = TF(closed_loop_numerator, closed_loop_denominator)
#     _, response, _ = lsim(closed_loop_model, U=ysp, T=t)

#     # return t, response, ysp
#         # PID Tuning (Ziegler-Nichols method as an example) Kp_fit, tau_fit, theta_fit
#     # Kc = 1.2 * (tau_fit / (Kp_fit * theta_fit))
#     # Ti = 2 * theta_fit
#     # Td = 0.5 * theta_fit
#     # Kc = 0.8 * (tau_fit / (Kp_fit * theta_fit))
#     # print(f"PID Parameters: Kc={Kc:.2f}, Ti={Ti:.2f}, Td={Td:.2f}")

#     # Simulate PID control
#     # response = simulate_pid_control(t, Kp_fit, tau_fit, theta_fit, Kc, TauI, Taud)

#     # Define possible parameter names in order
#     param_names_list = [["Kc", "TauI"], 
#                         ["Kc", "TauI", "Taud"], 
#                         ["Kc", "TauI", "Taud", "TauF"]]

#     # Select the correct names based on the tuple length
#     param_names = param_names_list[len(params) - 2]

#     # Create the annotation text dynamically
#     text = "<br>".join([f"{name} : {value:.2f}" for name, value in zip(param_names, params)])

#     annotation = dict(
#         x=0.95, 
#         y=0.4, 
#         xref="paper", 
#         yref="paper",
#         text=text,
#         showarrow=False
#     )

#     fig = go.Figure()

#     fig.add_trace(go.Scatter(x=t, y=ysp,
#                             mode='lines',
#                             name='Setpoint'))

#     fig.add_trace(go.Scatter(x=t, y=response,
#                             mode='lines',
#                             name='Response'))

#     # fig.add_trace(go.Scatter(x=[t0], y=[y0],
#     #                         mode='markers',
#     #                         marker=dict(color='red', size=10),
#     #                         name='t0, y0'))

#     # fig.add_trace(go.Scatter(x=[td], y=[y0],
#     #                         mode='markers',
#     #                         marker=dict(color='green', size=10),
#     #                         name='td, y0'))

#     # fig.add_trace(go.Scatter(x=[t1], y=[y_tau],
#     #                         mode='markers',
#     #                         marker=dict(color='blue', size=10),
#     #                         name='t1, y_tau'))

#     # fig.add_shape(go.layout.Shape(
#     #     type="line",
#     #     x0=min(t),
#     #     y0=y_inf,
#     #     x1=max(t),
#     #     y1=y_inf,
#     #     line=dict(color="purple", width=2, dash="dash"),
#     # ))

#     # fig.add_shape(go.layout.Shape(
#     #     type="line",
#     #     x0=min(t),
#     #     y0=y0,
#     #     x1=max(t),
#     #     y1=y0,
#     #     line=dict(color="orange", width=2, dash="dash"),
#     # ))

#     # fig.add_shape(go.layout.Shape(
#     #     type="line",
#     #     x0=min(t),
#     #     y0=y_tau,
#     #     x1=max(t),
#     #     y1=y_tau,
#     #     line=dict(color="gray", width=2, dash="dash"),
#     # ))


#     fig.update_layout(title=f'{controller_type} Controller',
#                     xaxis_title='Time',
#                     yaxis_title='Process Output',
#                     template="plotly_white",
#                     autosize=True,
#                     # annotations=[
#                     #     # dict(x=t0, y=y0, text=f"t0", showarrow=True, arrowhead=7),
#                     #     # dict(x=td, y=y0, text=f"td = {td:.2f}", showarrow=True, arrowhead=7),
#                     #     # dict(x=t1, y=y_tau, text=f"t1 = {t1:.2f}, y_tau = {y_tau:.2f}", showarrow=True, arrowhead=7),
#                     #     # dict(x=t1, y=y_inf, text=f"y_inf = {y_inf:.2f}", showarrow=True, arrowhead=7),
#                     #     dict(x=0.95, y=0.4, xref="paper", yref="paper",
#                     #         text=f"theta_est : {theta_est:.2f}<br>Kp_est: {Kp_est:.2f}<br>taun_est: {taun_est:.2f}<br>tau1_est: {tau1_est:.2f}<br>tau2_est: {tau2_est:.2f}",
#                     #         showarrow=False)
#                     # ]
#                     annotations = [annotation],
#                     )

#     # fig.show()  taun_est, tau1_est, tau2_est,  

#     # Convert the Plotly figure to an HTML div string
#     tuning_graph_html = fig.to_html(full_html=False)

#     return JsonResponse({'tuning_graph_html': tuning_graph_html})


@csrf_exempt
def simulate_pid(request):
    # if not hasattr(self, "pv") or self.pv is None:
    #     raise ValueError("Process variable (pv) is not defined. Perform data extraction first.")

    # # Use the identified process parameters
    # if not hasattr(self, "kp") or not hasattr(self, "tau") or not hasattr(self, "theta"):
    #     raise ValueError("Model parameters (Kp, Tau, Theta) are not defined. Perform identification first.")
    global controller_type
    controller_type = request.POST.get('controller_type')
    global criteria
    criteria  = request.POST.get('criteria')
    t = list(pv_data['x']), 
    t=t[0]
    process_data = np.array(pv_data['y'])
    OP = np.array(op_data['y'])

    
    Kp, Tau, theta = Kp_fit, tau_fit, theta_fit
    model1 = TF(Kp, [Tau, 1])
    model2 = TF([-0.5 * theta, 1], [0.5 * theta, 1])  # using first order Pade approximation for time delay """
    process_model = TF(np.polymul(model1.num,model2.num),np.polymul(model1.den,model2.den))

    # Retrieve time data and setpoint from pv and sp
    # time_data = np.array(self.pv.x)
    time_data = t #np.linspace(0, 50, 500)
    # Unit step change in setpoint
    ysp = np.ones_like(time_data)
    ysp[0] = 0
    initial_PID_params = [1.0, 1.0, 1.0]

    # Define closed loop transfer function,compute the response of the closed loop to a step change in setpoint and calculate ISE
    def objective_function(PID_params, process_model, time, ysp, criteria):
        Kp, Ki, Kd = PID_params
        PID = TF([Kp, Ki, Kd], [1, 0])  # Transfer fucntion for the controller is Kp+Ki*(1/s)+Kd*s
        closed_loop_numerator = np.polymul(process_model.num, PID.num)
        closed_loop_denominator = np.polyadd(np.polymul(process_model.den, PID.den),
                                                np.polymul(process_model.num, PID.num))
        closed_loop_model = TF(closed_loop_numerator, closed_loop_denominator)

        # compute response of close loop to step change in setpoint
        _, y, _ = lsim(closed_loop_model, U=ysp, T=time)
        offset = ysp - y
        if criteria == "ISE":
            ISE = np.sum(offset ** 2) * (time[1] - time[0])  # approximating the integral using Riemann sum
            obj = ISE
        elif criteria == "IAE":
            IAE = np.sum(np.absolute(offset)) * (time[1] - time[0])  # approximating the integral using Riemann sum
            obj = IAE
        elif criteria == "ITAE":
            ITAE = np.sum(time * np.absolute(offset)) * (
                        time[1] - time[0])  # approximating the integral using Riemann sum
            obj = ITAE
        return obj

    # ITAE
    if criteria == "ITAE":
        optm_result1 = minimize(objective_function, initial_PID_params, args=(process_model, time_data, ysp, criteria),
                                bounds=((0, None), (0, None), (0, None)), method='L-BFGS-B')
        PID_params_optimal = optm_result1.x
        Kp_optimal, Ki_optimal, Kd_optimal = PID_params_optimal

        # closed loop response with optimal PID controller
        PID_ITAE = TF([Kp_optimal, Ki_optimal, Kd_optimal], [1, 0])
        closed_loop_numerator_opt = np.polymul(process_model.num, PID_ITAE.num)
        closed_loop_denominator_opt = np.polyadd(np.polymul(process_model.den, PID_ITAE.den),
                                                    np.polymul(process_model.num, PID_ITAE.num))
        closed_loop_model_opt = TF(closed_loop_numerator_opt, closed_loop_denominator_opt)

        _, y_opt, _ = lsim(closed_loop_model_opt, U=ysp, T=time_data)

    # IAE
    if criteria == "IAE":
        optm_result2 = minimize(objective_function, initial_PID_params, args=(process_model, time_data, ysp, criteria),
                                bounds=((0, None), (0, None), (0, None)), method='L-BFGS-B')
        PID_params_optimal = optm_result2.x
        Kp_optimal, Ki_optimal, Kd_optimal = PID_params_optimal

        # closed loop response with optimal PID controller
        PID_IAE = TF([Kp_optimal, Ki_optimal, Kd_optimal], [1, 0])
        closed_loop_numerator_opt = np.polymul(process_model.num, PID_IAE.num)
        closed_loop_denominator_opt = np.polyadd(np.polymul(process_model.den, PID_IAE.den),
                                                    np.polymul(process_model.num, PID_IAE.num))
        closed_loop_model_opt = TF(closed_loop_numerator_opt, closed_loop_denominator_opt)

        _, y_opt, _ = lsim(closed_loop_model_opt, U=ysp, T=time_data)

    # ISE
    if criteria == "ISE":
        optm_result3 = minimize(objective_function, initial_PID_params, args=(process_model, time_data, ysp, criteria),
                                bounds=((0, None), (0, None), (0, None)), method='L-BFGS-B')
        PID_params_optimal = optm_result3.x
        Kp_optimal, Ki_optimal, Kd_optimal = PID_params_optimal

        # closed loop response with optimal PID controller
        PID_ISE = TF([Kp_optimal, Ki_optimal, Kd_optimal], [1, 0])
        closed_loop_numerator_opt = np.polymul(process_model.num, PID_ISE.num)
        closed_loop_denominator_opt = np.polyadd(np.polymul(process_model.den, PID_ISE.den),
                                                    np.polymul(process_model.num, PID_ISE.num))
        closed_loop_model_opt = TF(closed_loop_numerator_opt, closed_loop_denominator_opt)

        _, y_opt, _ = lsim(closed_loop_model_opt, U=ysp, T=time_data)

    # return time_data, y_opt, ysp, PID_params_optimal
    response = y_opt
    global params
    params = PID_params_optimal
    
    # return t, response, ysp
        # PID Tuning (Ziegler-Nichols method as an example) Kp_fit, tau_fit, theta_fit
    # Kc = 1.2 * (tau_fit / (Kp_fit * theta_fit))
    # Ti = 2 * theta_fit
    # Td = 0.5 * theta_fit
    # Kc = 0.8 * (tau_fit / (Kp_fit * theta_fit))
    # print(f"PID Parameters: Kc={Kc:.2f}, Ti={Ti:.2f}, Td={Td:.2f}")

    # Simulate PID control
    # response = simulate_pid_control(t, Kp_fit, tau_fit, theta_fit, Kc, TauI, Taud)

    # Define possible parameter names in order
    # param_names_list = [["Kc", "TauI"], 
    #                     ["Kc", "TauI", "Taud"], 
    #                     ["Kc", "TauI", "Taud", "TauF"]]
    param_names_list = [["Kp"], 
                        ["Kp", "Ki"], 
                        ["Kp", "Ki", "Kd"]]

    # Select the correct names based on the tuple length
    global param_names
    param_names = param_names_list[len(params) - 1]

    # Create the annotation text dynamically
    text = "<br>".join([f"{name} : {value:.2f}" for name, value in zip(param_names, params)])

    annotation = dict(
        x=0.95, 
        y=0.4, 
        xref="paper", 
        yref="paper",
        text=text,
        showarrow=False
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=ysp,
                            mode='lines',
                            name='Setpoint'))

    fig.add_trace(go.Scatter(x=t, y=response,
                            mode='lines',
                            name='Response'))

    # fig.add_trace(go.Scatter(x=[t0], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='red', size=10),
    #                         name='t0, y0'))

    # fig.add_trace(go.Scatter(x=[td], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='green', size=10),
    #                         name='td, y0'))

    # fig.add_trace(go.Scatter(x=[t1], y=[y_tau],
    #                         mode='markers',
    #                         marker=dict(color='blue', size=10),
    #                         name='t1, y_tau'))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_inf,
    #     x1=max(t),
    #     y1=y_inf,
    #     line=dict(color="purple", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y0,
    #     x1=max(t),
    #     y1=y0,
    #     line=dict(color="orange", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_tau,
    #     x1=max(t),
    #     y1=y_tau,
    #     line=dict(color="gray", width=2, dash="dash"),
    # ))


    fig.update_layout(title=f'{criteria} - {controller_type} Controller',
                    xaxis_title='Time',
                    yaxis_title='Process Output',
                    template="plotly_white",
                    autosize=True,
                    # annotations=[
                    #     # dict(x=t0, y=y0, text=f"t0", showarrow=True, arrowhead=7),
                    #     # dict(x=td, y=y0, text=f"td = {td:.2f}", showarrow=True, arrowhead=7),
                    #     # dict(x=t1, y=y_tau, text=f"t1 = {t1:.2f}, y_tau = {y_tau:.2f}", showarrow=True, arrowhead=7),
                    #     # dict(x=t1, y=y_inf, text=f"y_inf = {y_inf:.2f}", showarrow=True, arrowhead=7),
                    #     dict(x=0.95, y=0.4, xref="paper", yref="paper",
                    #         text=f"theta_est : {theta_est:.2f}<br>Kp_est: {Kp_est:.2f}<br>taun_est: {taun_est:.2f}<br>tau1_est: {tau1_est:.2f}<br>tau2_est: {tau2_est:.2f}",
                    #         showarrow=False)
                    # ]
                    annotations = [annotation],
                    )

    # fig.show()  taun_est, tau1_est, tau2_est,  

    # Convert the Plotly figure to an HTML div string
    global tuning_graph
    tuning_graph = fig
    tuning_graph_html = fig.to_html(full_html=False)

    return JsonResponse({'tuning_graph_html': tuning_graph_html})


@csrf_exempt
def simulate_pi(request):
    # if not hasattr(self, "pv") or self.pv is None:
    #     raise ValueError("Process variable (pv) is not defined. Perform data extraction first.")

    # # Use the identified process parameters
    # if not hasattr(self, "kp") or not hasattr(self, "tau") or not hasattr(self, "theta"):
    #     raise ValueError("Model parameters (Kp, Tau, Theta) are not defined. Perform identification first.")
    global controller_type
    controller_type = request.POST.get('controller_type')
    global criteria
    criteria  = request.POST.get('criteria')
    t = list(pv_data['x']), 
    t=t[0]
    # process_data = np.array(pv_data['y'])
    # OP = np.array(op_data['y'])
    
    Kp, Tau, theta = Kp_fit, tau_fit, theta_fit
    
    model1 = TF(Kp, [Tau, 1])
    model2 = TF([-0.5 * theta, 1], [0.5 * theta, 1])  # using first order Pade approximation for time delay
    process_model = TF(np.polymul(model1.num, model2.num), np.polymul(model1.den, model2.den))

    # Retrieve time data and setpoint from pv and sp
    # time_data = np.array(self.pv.x)
    time_data = t #np.linspace(0, 50, 500)
    # Unit step change in setpoint
    ysp = np.ones_like(time_data)
    ysp[0] = 0
    initial_PI_params = [1.0,1.0]

    # Define closed loop transfer function,compute the response of the closed loop to a step change in setpoint and calculate ISE
    def objective_function(PI_params, process_model, time, ysp, criteria):
        Kp, Ki = PI_params
        # PI = ctrl.TransferFunction([Kp,Ki],[1,0])
        PI = TF([Kp, Ki], [1, 0])  # Transfer fucntion for the controller is Kp+Ki*(1/s)
        closed_loop_numerator = np.polymul(process_model.num, PI.num)
        closed_loop_denominator = np.polyadd(np.polymul(process_model.den, PI.den),
                                                np.polymul(process_model.num, PI.num))
        closed_loop_model = TF(closed_loop_numerator, closed_loop_denominator)
        # closed_loop_model = ctrl.feedback(PI * process_model, 1)

        # compute response of close loop to step change in setpoint
        _, y, _ = lsim(closed_loop_model, U=ysp, T=time)
        offset = ysp - y
        if criteria == "ISE":
            ISE = np.sum(offset ** 2) * (time[1] - time[0])  # approximating the integral using Riemann sum
            obj = ISE
        elif criteria == "IAE":
            IAE = np.sum(np.absolute(offset)) * (time[1] - time[0])  # approximating the integral using Riemann sum
            obj = IAE
        elif criteria == "ITAE":
            ITAE = np.sum(time * np.absolute(offset)) * (
                        time[1] - time[0])  # approximating the integral using Riemann sum
            obj = ITAE
        return obj

    #################  ITAE  #####################
    if criteria == "ITAE":
        optm_result = minimize(objective_function, initial_PI_params, args=(process_model, time_data, ysp, criteria),
                                bounds=((0, None), (0, None)), method='L-BFGS-B')
        PI_params_optimal = optm_result.x
        Kp_optimal, Ki_optimal = PI_params_optimal
        print(f"Optimal PI parameters: Kp={Kp_optimal}, Ki={Ki_optimal}")

        # closed loop response with optimal PID controller
        PI_ISE = TF([Kp_optimal, Ki_optimal], [1, 0])  # TF([Kp_optimal,Ki_optimal],[1,0])
        closed_loop_numerator_opt = np.polymul(process_model.num, PI_ISE.num)
        closed_loop_denominator_opt = np.polyadd(np.polymul(process_model.den, PI_ISE.den),
                                                    np.polymul(process_model.num, PI_ISE.num))
        closed_loop_model_opt = TF(closed_loop_numerator_opt, closed_loop_denominator_opt)
        _, y_opt, _ = lsim(closed_loop_model_opt, U=ysp, T=time_data)  # lsim(closed_loop_model_opt,U = ysp,T=time)

    #################  IAE  #####################
    if criteria == "IAE":
        optm_result = minimize(objective_function, initial_PI_params, args=(process_model, time_data, ysp, criteria),
                                bounds=((0, None), (0, None)), method='L-BFGS-B')
        PI_params_optimal = optm_result.x
        Kp_optimal, Ki_optimal = PI_params_optimal
        print(f"Optimal PI parameters: Kp={Kp_optimal}, Ki={Ki_optimal}")

        PI_ISE = TF([Kp_optimal, Ki_optimal], [1, 0])
        closed_loop_numerator_opt = np.polymul(process_model.num, PI_ISE.num)
        closed_loop_denominator_opt = np.polyadd(np.polymul(process_model.den, PI_ISE.den),
                                                    np.polymul(process_model.num, PI_ISE.num))
        closed_loop_model_opt = TF(closed_loop_numerator_opt, closed_loop_denominator_opt)
        _, y_opt, _ = lsim(closed_loop_model_opt, U=ysp, T=time_data)

    #################  ISE  #####################
    if criteria == "ISE":
        optm_result = minimize(objective_function, initial_PI_params, args=(process_model, time_data, ysp, criteria),
                                bounds=((0, None), (0, None)), method='L-BFGS-B')
        PI_params_optimal = optm_result.x
        Kp_optimal, Ki_optimal = PI_params_optimal
        print(f"Optimal PI parameters: Kp={Kp_optimal}, Ki={Ki_optimal}")

        PI_ISE = TF([Kp_optimal, Ki_optimal], [1, 0])
        closed_loop_numerator_opt = np.polymul(process_model.num, PI_ISE.num)
        closed_loop_denominator_opt = np.polyadd(np.polymul(process_model.den, PI_ISE.den),
                                                    np.polymul(process_model.num, PI_ISE.num))
        closed_loop_model_opt = TF(closed_loop_numerator_opt, closed_loop_denominator_opt)

        _, y_opt, _ = lsim(closed_loop_model_opt, U=ysp, T=time_data)

    global params
    # return time_data, y_opt, ysp, PI_params_optimal
    params = PI_params_optimal
    response = y_opt
    # Define possible parameter names in order
    # param_names_list = [["Kc", "TauI"], 
    #                     ["Kc", "TauI", "Taud"], 
    #                     ["Kc", "TauI", "Taud", "TauF"]]
    param_names_list = [["Kp"], 
                        ["Kp", "Ki"], 
                        ["Kp", "Ki", "Kd"]]

    # Select the correct names based on the tuple length
    global param_names
    param_names = param_names_list[len(params) - 1]

    # Create the annotation text dynamically
    text = "<br>".join([f"{name} : {value:.2f}" for name, value in zip(param_names, params)])

    annotation = dict(
        x=0.95, 
        y=0.4, 
        xref="paper", 
        yref="paper",
        text=text,
        showarrow=False
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=ysp,
                            mode='lines',
                            name='Setpoint'))

    fig.add_trace(go.Scatter(x=t, y=response,
                            mode='lines',
                            name='Response'))

    # fig.add_trace(go.Scatter(x=[t0], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='red', size=10),
    #                         name='t0, y0'))

    # fig.add_trace(go.Scatter(x=[td], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='green', size=10),
    #                         name='td, y0'))

    # fig.add_trace(go.Scatter(x=[t1], y=[y_tau],
    #                         mode='markers',
    #                         marker=dict(color='blue', size=10),
    #                         name='t1, y_tau'))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_inf,
    #     x1=max(t),
    #     y1=y_inf,
    #     line=dict(color="purple", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y0,
    #     x1=max(t),
    #     y1=y0,
    #     line=dict(color="orange", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_tau,
    #     x1=max(t),
    #     y1=y_tau,
    #     line=dict(color="gray", width=2, dash="dash"),
    # ))


    fig.update_layout(title=f'{criteria} - {controller_type} Controller',
                    xaxis_title='Time',
                    yaxis_title='Process Output',
                    template="plotly_white",
                    autosize=True,
                    # annotations=[
                    #     # dict(x=t0, y=y0, text=f"t0", showarrow=True, arrowhead=7),
                    #     # dict(x=td, y=y0, text=f"td = {td:.2f}", showarrow=True, arrowhead=7),
                    #     # dict(x=t1, y=y_tau, text=f"t1 = {t1:.2f}, y_tau = {y_tau:.2f}", showarrow=True, arrowhead=7),
                    #     # dict(x=t1, y=y_inf, text=f"y_inf = {y_inf:.2f}", showarrow=True, arrowhead=7),
                    #     dict(x=0.95, y=0.4, xref="paper", yref="paper",
                    #         text=f"theta_est : {theta_est:.2f}<br>Kp_est: {Kp_est:.2f}<br>taun_est: {taun_est:.2f}<br>tau1_est: {tau1_est:.2f}<br>tau2_est: {tau2_est:.2f}",
                    #         showarrow=False)
                    # ]
                    annotations = [annotation],
                    )

    # fig.show()  taun_est, tau1_est, tau2_est,  
    global tuning_graph
    tuning_graph = fig
    # Convert the Plotly figure to an HTML div string
    tuning_graph_html = fig.to_html(full_html=False)

    return JsonResponse({'tuning_graph_html': tuning_graph_html})


@csrf_exempt
def simulate_p(request):
    # if not hasattr(self, "pv") or self.pv is None:
    #     raise ValueError("Process variable (pv) is not defined. Perform data extraction first.")

    # # Use the identified process parameters
    # if not hasattr(self, "kp") or not hasattr(self, "tau") or not hasattr(self, "theta"):
    #     raise ValueError("Model parameters (Kp, Tau, Theta) are not defined. Perform identification first.")
    global controller_type
    controller_type = request.POST.get('controller_type')
    global criteria
    criteria  = request.POST.get('criteria')
    t = list(pv_data['x']), 
    t=t[0]
    # process_data = np.array(pv_data['y'])
    # OP = np.array(op_data['y'])
    
    Kp, Tau, theta = Kp_fit, tau_fit, theta_fit
    
    model1 = TF(Kp, [Tau, 1])
    model2 = TF([-0.5 * theta, 1], [0.5 * theta, 1])  # using first order Pade approximation for time delay

    process_model = TF(np.polymul(model1.num, model2.num), np.polymul(model1.den, model2.den))
    # Retrieve time data and setpoint from pv and sp
    # time_data = np.array(self.pv.x)
    time_data = t #np.linspace(0, 50, 500)
    # Unit step change in setpoint
    ysp = np.ones_like(time_data)
    ysp[0] = 0
    initial_P_params = [1.0]

    # Define closed loop transfer function,compute the response of the closed loop to a step change in setpoint and calculate ISE
    def objective_function(P_params, process_model, time, ysp, criteria):
        Kp = P_params
        P = TF([0, Kp], [0, 1])  # Transfer fucntion for the controller is u(s) = Kp*e(s)
        closed_loop_numerator = np.polymul(process_model.num.ravel(), P.num.ravel())
        closed_loop_denominator = np.polyadd(np.polymul(process_model.den.ravel(), P.den.ravel()),
                                                np.polymul(process_model.num.ravel(), P.num.ravel())).ravel()
        closed_loop_model = TF(closed_loop_numerator, closed_loop_denominator)

        # compute response of close loop to step change in setpoint
        _, y, _ = lsim(closed_loop_model, U=ysp, T=time)
        offset = ysp - y
        if criteria == "ISE":
            ISE = np.sum(offset ** 2) * (time[1] - time[0])  # approximating the integral using Riemann sum
            obj = ISE
        elif criteria == "IAE":
            IAE = np.sum(np.absolute(offset)) * (time[1] - time[0])  # approximating the integral using Riemann sum
            obj = IAE
        elif criteria == "ITAE":
            ITAE = np.sum(time * np.absolute(offset)) * (
                        time[1] - time[0])  # approximating the integral using Riemann sum
            obj = ITAE
        return obj

    ################## IAE #######################
    if criteria == "IAE":
        optm_result = minimize(objective_function, initial_P_params, args=(process_model, time_data, ysp, criteria),
                                bounds=[(0, None)], method='L-BFGS-B')
        P_params_optimal = optm_result.x
        Kp_optimal = P_params_optimal
        print(f"Optimal P parameters: Kp={Kp_optimal}")

        # closed loop response with optimal P controller
        P_ISE = TF([0, Kp_optimal], [0, 1])
        closed_loop_numerator_opt = np.polymul(process_model.num.ravel(), P_ISE.num.ravel())
        closed_loop_denominator_opt = np.polyadd(np.polymul(process_model.den.ravel(), P_ISE.den.ravel()),
                                                    np.polymul(process_model.num.ravel(), P_ISE.num.ravel()))
        closed_loop_model_opt = TF(closed_loop_numerator_opt, closed_loop_denominator_opt)

        _, y_opt, _ = lsim(closed_loop_model_opt, U=ysp, T=time_data)

    ########### ISE  ###################
    elif criteria == "ISE":
        optm_result = minimize(objective_function, initial_P_params, args=(process_model, time_data, ysp, criteria),
                                bounds=[(0, None)], method='L-BFGS-B')
        P_params_optimal = optm_result.x
        Kp_optimal = P_params_optimal
        print(f"Optimal P parameters: Kp={Kp_optimal}")

        # closed loop response with optimal P controller
        P_ISE = TF([0, Kp_optimal], [0, 1])
        closed_loop_numerator_opt = np.polymul(process_model.num.ravel(), P_ISE.num.ravel())
        closed_loop_denominator_opt = np.polyadd(np.polymul(process_model.den.ravel(), P_ISE.den.ravel()),
                                                    np.polymul(process_model.num.ravel(), P_ISE.num.ravel()))
        closed_loop_model_opt = TF(closed_loop_numerator_opt, closed_loop_denominator_opt)

        _, y_opt, _ = lsim(closed_loop_model_opt, U=ysp, T=time_data)

    ############### ITAE  ###########################
    if criteria == "ITAE":
        optm_result = minimize(objective_function, initial_P_params, args=(process_model, time_data, ysp, criteria),
                                bounds=[(0, None)], method='L-BFGS-B')
        P_params_optimal = optm_result.x
        Kp_optimal = P_params_optimal
        print(f"Optimal P parameters: Kp={Kp_optimal}")

        # closed loop response with optimal P controller
        P_ISE = TF([0, Kp_optimal], [0, 1])
        closed_loop_numerator_opt = np.polymul(process_model.num.ravel(), P_ISE.num.ravel())
        closed_loop_denominator_opt = np.polyadd(np.polymul(process_model.den.ravel(), P_ISE.den.ravel()),
                                                    np.polymul(process_model.num.ravel(), P_ISE.num.ravel()))
        closed_loop_model_opt = TF(closed_loop_numerator_opt, closed_loop_denominator_opt)

        _, y_opt, _ = lsim(closed_loop_model_opt, U=ysp, T=time_data)

    # return time_data, y_opt, ysp, P_params_optimal
    global params
    params = P_params_optimal
    response = y_opt
    # Define possible parameter names in order
    # param_names_list = [["Kc", "TauI"], 
    #                     ["Kc", "TauI", "Taud"], 
    #                     ["Kc", "TauI", "Taud", "TauF"]]
    param_names_list = [["Kp"], 
                        ["Kp", "Ki"], 
                        ["Kp", "Ki", "Kd"]]

    # Select the correct names based on the tuple length
    global param_names
    param_names = param_names_list[len(params) - 1]

    
    # Create the annotation text dynamically
    text = "<br>".join([f"{name} : {value:.2f}" for name, value in zip(param_names, params)])

    annotation = dict(
        x=0.95, 
        y=0.4, 
        xref="paper", 
        yref="paper",
        text=text,
        showarrow=False
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=ysp,
                            mode='lines',
                            name='Setpoint'))

    fig.add_trace(go.Scatter(x=t, y=response,
                            mode='lines',
                            name='Response'))

    # fig.add_trace(go.Scatter(x=[t0], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='red', size=10),
    #                         name='t0, y0'))

    # fig.add_trace(go.Scatter(x=[td], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='green', size=10),
    #                         name='td, y0'))

    # fig.add_trace(go.Scatter(x=[t1], y=[y_tau],
    #                         mode='markers',
    #                         marker=dict(color='blue', size=10),
    #                         name='t1, y_tau'))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_inf,
    #     x1=max(t),
    #     y1=y_inf,
    #     line=dict(color="purple", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y0,
    #     x1=max(t),
    #     y1=y0,
    #     line=dict(color="orange", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_tau,
    #     x1=max(t),
    #     y1=y_tau,
    #     line=dict(color="gray", width=2, dash="dash"),
    # ))


    fig.update_layout(title=f'{criteria} - {controller_type} Controller',
                    xaxis_title='Time',
                    yaxis_title='Process Output',
                    template="plotly_white",
                    autosize=True,
                    # annotations=[
                    #     # dict(x=t0, y=y0, text=f"t0", showarrow=True, arrowhead=7),
                    #     # dict(x=td, y=y0, text=f"td = {td:.2f}", showarrow=True, arrowhead=7),
                    #     # dict(x=t1, y=y_tau, text=f"t1 = {t1:.2f}, y_tau = {y_tau:.2f}", showarrow=True, arrowhead=7),
                    #     # dict(x=t1, y=y_inf, text=f"y_inf = {y_inf:.2f}", showarrow=True, arrowhead=7),
                    #     dict(x=0.95, y=0.4, xref="paper", yref="paper",
                    #         text=f"theta_est : {theta_est:.2f}<br>Kp_est: {Kp_est:.2f}<br>taun_est: {taun_est:.2f}<br>tau1_est: {tau1_est:.2f}<br>tau2_est: {tau2_est:.2f}",
                    #         showarrow=False)
                    # ]
                    annotations = [annotation],
                    )

    # fig.show()  taun_est, tau1_est, tau2_est,  
    global tuning_graph
    tuning_graph = fig
    # Convert the Plotly figure to an HTML div string
    tuning_graph_html = fig.to_html(full_html=False)

    return JsonResponse({'tuning_graph_html': tuning_graph_html})


# Design of P, PI and PID controllers for FOPTD, IPTD and SOPTD using ISE, IAE and ITAE
# The code asks the user to input his selection of crtierion (ISE, IAE, ITAE) and process model type (FOPTD, IPDT, SOPTD)


# User Input for criteria and process model
# criteria = input("Enter criteria (ISE, IAE, ITAE): ")
#input("Enter process model (FOPTD, IPDT, SOPTD): ")
@csrf_exempt
def simulate_close_loop_response(request):

    global controller_type
    controller_type = request.POST.get('controller_type')
    global criteria
    criteria  = request.POST.get('criteria')

    process_model = modeling_type 

    # Define simulation parameters
    # dt = 0.01  # Discrete time step
    dt = 0.1
    # t_final = 25
    # time = np.arange(0, t_final, dt)  # Time array
    t = list(pv_data['x']), 
    t=t[0]
    time = t
    process_params = modeling_params

    # # Define process model parameters based on user selection
    # if process_model == "First-Order Plus Dead Time (FOPDT)":
    #     Kp = 1.0  # Process gain
    #     Tau = 2.0  # Time constant
    #     theta = 1.0  # Time delay in seconds
    #     process_params = [Kp, Tau, theta]
    # elif process_model == "IPDT":
    #     Kp = 1.0
    #     theta = 1.0
    #     process_params = [Kp, theta]
    # elif process_model == "SOPTD":
    #     Kp = 1.0
    #     Tau = 2.0
    #     Zeta = 0.7
    #     theta = 1.0
    #     process_params = [Kp, Tau, Zeta, theta]
    # else:
    #     raise ValueError("Invalid process model selected!")

    # Ziegler-Nichols tuning method (only for FOPTD)
    def ziegler_nichols_tuning(Kp, Tau, theta):
        Ku = (0.6 * Tau) / (Kp * theta)
        Tu = 2 * theta  # Approximate ultimate period
        return {
            "P": [0.5 * Ku, 0.0, 0.0],
            "PI": [0.45 * Ku, 0.54 * Ku / Tu, 0.0],
            "PID": [0.6 * Ku, 1.2 * Ku / Tu, 0.075 * Ku * Tu]
        }

    # Custom heuristic initialization for SOPTD and IPDT
    custom_initial_params = {
        "P": [1.0, 0.0, 0.0],
        "PI": [1.0, 1.0, 0.0],
        "PID": [1.0, 1.0, 1.0]
    }

    # Closed-loop response function
    def closed_loop_response(PID_params, process_params, time, ysp, controller_type):
        Kc, Ki, Kd = PID_params

        if process_model == "First-Order Plus Dead Time (FOPDT)":
            Kp, Tau, theta = process_params
        elif process_model == "Integrator Plus Dead Time (IPDT)":
            Kp, theta = process_params
            Tau = dt  # Use discrete integration step for proper numerical stability
        elif process_model == "Second-Order Plus Dead Time (SOPDT)":
            Kp, Tau, Zeta, theta = process_params

        y = np.zeros_like(time)
        u = np.zeros_like(time)
        e = np.zeros_like(time)
        integral = 0.0
        prev_error = 0.0
        delay_steps = int(theta / dt)
        u_delayed = deque([0] * delay_steps, maxlen=delay_steps)

        for i in range(1, len(time)):
            e[i] = ysp[i] - y[i - 1]
            integral = cumulative_trapezoid(e[:i+1], time[:i+1], initial=0)[-1]

            if i > 1:
                derivative = np.gradient(e[:i+1], dt)[-1]
            else:
                derivative = 0

            derivative_filtered = 0.9 * derivative + 0.1 * prev_error

            if controller_type == "P":
                u[i] = Kc * e[i]
            elif controller_type == "PI":
                u[i] = Kc * e[i] + Ki * integral
            elif controller_type == "PID":
                u[i] = Kc * e[i] + Ki * integral + Kd * derivative_filtered

            prev_error = e[i]

            u_delayed.append(u[i])
            if process_model == "First-Order Plus Dead Time (FOPDT)":
                y[i] = y[i - 1] + (dt / Tau) * (Kp * u_delayed[0] - y[i - 1])
            elif process_model == "Integrator Plus Dead Time (IPDT)":
                y[i] = y[i - 1] + dt * Kp * u_delayed[0]  # Proper integration for IPDT
            elif process_model == "Second-Order Plus Dead Time (SOPDT)":
                y[i] = y[i - 1] + (dt / (Tau**2)) * (Kp * u_delayed[0] - 2 * Zeta * Tau * y[i-1] - y[i - 2])

        return y

    # Define objective function
    def objective_function(PID_params, process_params, time, ysp, controller_type):
        y = closed_loop_response(PID_params, process_params, time, ysp, controller_type)
        offset = ysp - y

        if criteria == "ISE":
            return np.sum(offset**2) * dt  # Integral of Squared Error
        elif criteria == "IAE":
            return np.sum(np.abs(offset)) * dt  # Integral of Absolute Error
        elif criteria == "ITAE":
            return np.sum(time * np.abs(offset)) * dt  # Integral of Time-weighted Absolute Error
        else:
            raise ValueError("Invalid criteria selected!")

    # Solve the optimization problem
    ysp = np.ones_like(time)
    ysp[0] = 0

    # Get initial parameters
    if process_model == "First-Order Plus Dead Time (FOPDT)":
        initial_params_dict = ziegler_nichols_tuning(Kp_fit, tau_fit, theta_fit)
    else:
        initial_params_dict = custom_initial_params

    # Optimize for P, PI, and PID using L-BFGS-B
    # controller_types = ["P", "PI", "PID"]
    optimal_params = {}

    # for controller_type in controller_types:
    initial_params = initial_params_dict[controller_type]
    if controller_type == "P":
        bounds = [(0, 10), (0, 0), (0, 0)]
    elif controller_type == "PI":
        bounds = [(0, 10), (0, 10), (0, 0)]
    elif controller_type == "PID":
        bounds = [(0, 10), (0, 10), (0, 10)]

    result = minimize(objective_function, initial_params, args=(process_params, time, ysp, controller_type), method='L-BFGS-B', bounds=bounds)
    optimal_params[controller_type] = result.x
    print(f"Optimal {controller_type} Controller: Kp={result.x[0]}, Ki={result.x[1]}, Kd={result.x[2]}")

    # # Plot closed-loop responses for all controllers
    # plt.figure(figsize=(10, 6))
    # for controller_type in controller_types:
    y_opt = closed_loop_response(optimal_params[controller_type], process_params, time, ysp, controller_type)
    
    global params
    params = optimal_params[controller_type]
    response = y_opt
    # Define possible parameter names in order
    # param_names_list = [["Kc", "TauI"], 
    #                     ["Kc", "TauI", "Taud"], 
    #                     ["Kc", "TauI", "Taud", "TauF"]]
    param_names_list = [["Kp"], 
                        ["Kp", "Ki"], 
                        ["Kp", "Ki", "Kd"]]

    # Select the correct names based on the tuple length
    global param_names
    param_names = param_names_list[len(params) - 1]

    print(params)
    # Create the annotation text dynamically
    text = "<br>".join([f"{name} : {value:.2f}" for name, value in zip(param_names, params)])

    annotation = dict(
        x=0.95, 
        y=0.4, 
        xref="paper", 
        yref="paper",
        text=text,
        showarrow=False
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=ysp,
                            mode='lines',
                            name='Setpoint'))

    fig.add_trace(go.Scatter(x=t, y=response,
                            mode='lines',
                            name='Response'))

    # fig.add_trace(go.Scatter(x=[t0], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='red', size=10),
    #                         name='t0, y0'))

    # fig.add_trace(go.Scatter(x=[td], y=[y0],
    #                         mode='markers',
    #                         marker=dict(color='green', size=10),
    #                         name='td, y0'))

    # fig.add_trace(go.Scatter(x=[t1], y=[y_tau],
    #                         mode='markers',
    #                         marker=dict(color='blue', size=10),
    #                         name='t1, y_tau'))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_inf,
    #     x1=max(t),
    #     y1=y_inf,
    #     line=dict(color="purple", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y0,
    #     x1=max(t),
    #     y1=y0,
    #     line=dict(color="orange", width=2, dash="dash"),
    # ))

    # fig.add_shape(go.layout.Shape(
    #     type="line",
    #     x0=min(t),
    #     y0=y_tau,
    #     x1=max(t),
    #     y1=y_tau,
    #     line=dict(color="gray", width=2, dash="dash"),
    # ))


    fig.update_layout(title=f'{criteria} - {controller_type} Controller',
                    xaxis_title='Time',
                    yaxis_title='Process Output',
                    template="plotly_white",
                    autosize=True,
                    # annotations=[
                    #     # dict(x=t0, y=y0, text=f"t0", showarrow=True, arrowhead=7),
                    #     # dict(x=td, y=y0, text=f"td = {td:.2f}", showarrow=True, arrowhead=7),
                    #     # dict(x=t1, y=y_tau, text=f"t1 = {t1:.2f}, y_tau = {y_tau:.2f}", showarrow=True, arrowhead=7),
                    #     # dict(x=t1, y=y_inf, text=f"y_inf = {y_inf:.2f}", showarrow=True, arrowhead=7),
                    #     dict(x=0.95, y=0.4, xref="paper", yref="paper",
                    #         text=f"theta_est : {theta_est:.2f}<br>Kp_est: {Kp_est:.2f}<br>taun_est: {taun_est:.2f}<br>tau1_est: {tau1_est:.2f}<br>tau2_est: {tau2_est:.2f}",
                    #         showarrow=False)
                    # ]
                    annotations = [annotation],
                    )

    # fig.show()  taun_est, tau1_est, tau2_est,  
    global tuning_graph
    tuning_graph = fig
    # Convert the Plotly figure to an HTML div string
    tuning_graph_html = fig.to_html(full_html=False)

    return JsonResponse({'tuning_graph_html': tuning_graph_html})
    # plt.plot(time, y_opt, label=f"{controller_type} Controller")

    # plt.plot(time, ysp, color="r", linestyle="--", label="Setpoint")
    # plt.title(f"Closed-loop response ({criteria}) for {process_model} model")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Response")
    # plt.legend()
    # plt.grid()
    # plt.show()





@csrf_exempt
def get_variables_report(request):
    modeling_graph_report = modeling_graph.to_html(full_html=False)
    tuning_graph_report = tuning_graph.to_html(full_html=False)
    return JsonResponse({'x_min':x_min,
                         'x_max':x_max,
                         'y_min':y_min,
                         'y_max':y_max,
                         'modeling_graph': modeling_graph_report,
                         'modeling_type':modeling_type,
                         'modeling_params':modeling_params.tolist(),
                         'modeling_params_names':modeling_params_names,
                        #  'theta_fit':theta_fit, 
                        #  'Kp_fit':Kp_fit, 
                        #  'tau_fit':tau_fit, 
                         'tuning_graph': tuning_graph_report,
                         'params':params.tolist(),
                         'param_names':param_names,
                         'tuning_criteria':criteria,
                         'controller_type':controller_type,
                         
                         })


@csrf_exempt
def modeling(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        input_value = data.get('input')

        # Call modeling function with input
        model_output = "Model result for " + input_value

        return JsonResponse({'model_output': model_output})

@csrf_exempt
def tuning(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        input_value = data.get('input')

        # Call tuning function with input
        tuning_output = "Tuning result for " + input_value

        return JsonResponse({'tuning_output': tuning_output})

@csrf_exempt
def report(request):
    # Generate a report based on previous results
    report_content = "Report generated..."
    return JsonResponse({'report': report_content})
