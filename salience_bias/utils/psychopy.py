import numpy as np

monitor_params = {
    'width_pix'    : 1920,
    'width_cm'     : 52.2,
    'view_dist_cm' : 34.0,
    'size_pix'     : [1920, 1080],
}

def pix2deg(pixels, monitor_params=monitor_params):
    # pix2deg(1) : 0.04098239784616305 (measured from experiment computer)
    data_pix = np.asarray(pixels)
    data_cm  = data_pix * (monitor_params['width_cm'] / monitor_params['width_pix'])
    data_deg = data_cm * (180.0 / np.pi) / monitor_params['view_dist_cm']
    return data_deg

def el2deg(x_vec, y_vec, monitor_params=monitor_params):
    x_vec = np.asarray(x_vec)
    y_vec = np.asarray(y_vec)
    x_vec = pix2deg( x_vec - monitor_params['size_pix'][0] / 2. )
    y_vec = pix2deg( monitor_params['size_pix'][1] / 2. - y_vec )
    # x_vec = 0.04098239784616305 * ( x_vec - monitor_params['size_pix'][0] / 2 )
    # y_vec = 0.04098239784616305 * ( monitor_params['size_pix'][1] / 2 - y_vec )
    return x_vec, y_vec
