import config
from models import LPP, NETT

##################### MODEL SELECTION
train_NETT = False
train_FBPLPP = True

##################### PARAMETER INITIALIZATION
# Define dataset
dataset = "Mayo"  # Mayo, COULE

# Define additional problem parameters
noise_level = 0.01  # Mayo 0.001, COULE 0.01

# Load configuration
cfg = config.initialize_default_config(dataset)
cfg = config.parse_config(cfg)

################## NETT Testing
if train_NETT:
    NETT_model = NETT.NETT(cfg)
    NETT_model.train(batch_size=8, n_epochs=100, noise_level=0.01, p=0.5)

################## FBP-LPP Testing
if train_FBPLPP:
    FBPLPP_model = LPP.FBP_LPP(cfg)
    FBPLPP_model.train(batch_size=8, n_epochs=100, noise_level=0.01)
