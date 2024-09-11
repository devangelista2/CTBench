import matplotlib.pyplot as plt
from torchvision import transforms

import config
from miscellanous import datasets, metrics, utilities
from models import FISTA, LPP, NETT

##################### TEST SETUP
NETT_test = True
FBPLPP_test = False
FISTA_test = False

##################### OUTPUT PARAMETERS
output_path = "./results"
output_metrics = []

SAVE_OUTPUT_IMAGE = True
PRINT_METRICS = True

##################### PARAMETER INITIALIZATION
# Define dataset
dataset = "Mayo" #Mayo, COULE
idx = 10 # 10 or 6

# Define additional problem parameters
noise_level = 0.001 #Mayo 0.001, COULE 0.01
 
# Load configuration
cfg = config.initialize_default_config(dataset)
cfg = config.parse_config(cfg)

img_ch, nx, ny = cfg["image_shape"]
gt_path = f"../data/{dataset}/test/"
device = cfg["device"]

# Load test data and x_true
test_data = datasets.ImageDataset(gt_path, transforms=transforms.Resize((nx, ny)))
x_true = test_data[idx]

# Initialize test problem
K = utilities.initialize_CT_projector(cfg)

y = K(x_true)
y_delta = y + utilities.gaussian_noise(y, noise_level)

################## NETT Testing
if NETT_test:
    NETT_model = NETT.NETT(cfg)
    NETT_model.load_weights()

    x_NETT = NETT_model(
        y_delta, lmbda=1e-4, x_true=x_true, step_size_F=None, step_size_R=1e-2, maxit=300
    ).reshape((nx, ny))

    if SAVE_OUTPUT_IMAGE:
        fname = f"{output_path}/NETT_{dataset}_{nx}_{cfg['angular_range']}_{cfg['n_angles']}.png"
        plt.imsave(fname, x_NETT.reshape(nx, ny), cmap='gray')

    if PRINT_METRICS:
        RE = metrics.RE(x_NETT.numpy(), x_true.numpy().reshape((nx, ny)))
        RMSE = metrics.RMSE(x_NETT.numpy(), x_true.numpy().reshape((nx, ny)))
        SSIM = metrics.SSIM(x_NETT.numpy(), x_true.numpy().reshape((nx, ny)))
        output_metrics.append(['NETT', RE, RMSE, SSIM])
        print(f"NETT: RE = {RE:0.4f}, RMSE: {RMSE:0.4f}, SSIM: {SSIM:0.4f}.")

################## FBP-LPP Testing
if FBPLPP_test:
    FBPLPP_model = LPP.FBP_LPP(cfg)
    FBPLPP_model.load_weights()

    x_FBPLPP = FBPLPP_model(y_delta).reshape((nx, ny))

    if SAVE_OUTPUT_IMAGE:
        fname = f"{output_path}/FBPLPP_{dataset}_{nx}_{cfg['angular_range']}_{cfg['n_angles']}.png"
        plt.imsave(fname, x_FBPLPP.reshape(nx, ny), cmap='gray')

    if PRINT_METRICS:
        RE = metrics.RE(x_FBPLPP.numpy(), x_true.numpy().reshape((nx, ny)))
        RMSE = metrics.RMSE(x_FBPLPP.numpy(), x_true.numpy().reshape((nx, ny)))
        SSIM = metrics.SSIM(x_FBPLPP.numpy(), x_true.numpy().reshape((nx, ny)))
        output_metrics.append(['FBP-LPP', RE, RMSE, SSIM])
   

################## FISTA-W Testing
if FISTA_test:
    FISTA_model = FISTA.FISTAWavelet(cfg)

    x_FISTA = FISTA_model(y_delta, lmbda=0.001, x_true=x_true, maxit=500).reshape((nx, ny))

    if SAVE_OUTPUT_IMAGE:
        fname = f"{output_path}/FISTA_{dataset}_{nx}_{cfg['angular_range']}_{cfg['n_angles']}.png"
        plt.imsave(fname, x_FISTA.reshape(nx, ny), cmap='gray')

    if PRINT_METRICS:
        RE = metrics.RE(x_FISTA, x_true.numpy().reshape((nx, ny)))
        RMSE = metrics.RMSE(x_FISTA, x_true.numpy().reshape((nx, ny)))
        SSIM = metrics.SSIM(x_FISTA, x_true.numpy().reshape((nx, ny)))
        output_metrics.append(['FBP-LPP', RE, RMSE, SSIM])
        print(f"FISTA-W: RE = {RE:0.4f}, RMSE: {RMSE:0.4f}, SSIM: {SSIM:0.4f}.")

################## Printing out
# if PRINT_METRICS:
#     print("**************************")
#     for metric in output_metrics:
#         print(f"{metric[0]}: RE = {metric[1]:0.4f}, RMSE: {metric[2]:0.4f}, SSIM: {metric[3]:0.4f}.")
#     print("**************************")