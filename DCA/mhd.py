# import sys
# sys.path.append('/mnt/disk1/dnkhanh/Multimodal/FLMultimodal/DCA')
import numpy as np
from dca.DCA import DCA
from dca.schemes import (
    DCALoggers,
    DelaunayGraphParams,
    ExperimentDirs,
    GeomCAParams,
    HDBSCANParams,
    REData,
)
import typer

# Set path to the output folders
experiment_path = "output/first_example"
experiment_id = "template_id1"

# Load data
R = np.load("/mnt/disk1/dnkhanh/Multimodal/FLMultimodal/fedtask/mhd_classification_cnum50_dist0_skew0_seed0/representations/mm_mhd_fedavg_Mmm_R20_B64_E2_CW0.00_CT1.00_LR0.0500_P1.00_S1234_LD-0.002_WD0.000_AVLIDL_CNIDL_CPIDL_TIDL/Round20_image+sound.npy")
E = np.load("/mnt/disk1/dnkhanh/Multimodal/FLMultimodal/fedtask/mhd_classification_cnum50_dist0_skew0_seed0/representations/mm_mhd_fedavg_Mmm_R20_B64_E2_CW0.00_CT1.00_LR0.0500_P1.00_S1234_LD-0.002_WD0.000_AVLIDL_CNIDL_CPIDL_TIDL/Round20_image.npy")

# Generate input parameters
data_config = REData(R=R, E=E)
experiment_config = ExperimentDirs(
    experiment_dir=experiment_path,
    experiment_id=experiment_id,
)
graph_config = DelaunayGraphParams()
hdbscan_config = HDBSCANParams()
geomCA_config = GeomCAParams()

# Initialize loggers
exp_loggers = DCALoggers(experiment_config.logs_dir)

# Run DCA
dca = DCA(
    experiment_config,
    graph_config,
    hdbscan_config,
    geomCA_config,
    loggers=exp_loggers,
)

dca_scores = dca.fit(data_config)