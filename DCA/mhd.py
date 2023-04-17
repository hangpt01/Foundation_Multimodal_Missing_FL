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
import os
import sys
sys.path.append("..")
from torch.utils.data import DataLoader
from benchmark.mhd_classification.model.mm import Model
from benchmark.mhd_classification.dataset import MHDDataset
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
testset = MHDDataset(
    root='../benchmark/RAW_DATA/MHD',
    train=False,
    download=True
)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)
all_modal = ['image', 'trajectory']
all_combin_key = ['trajectory', 'image+trajectory']
import torch
device = torch.device('cuda')
model = Model()
round_id = 500
prefix = "/mnt/disk1/dnkhanh/Multimodal/FLMultimodal/fedtask/mhd_classification_cnum50_dist0_skew0_seed0_image+trajectory_"
# Set path to the output folders
for setting in ["10+10+30", "0+0+50", "0+25+25"]:
    takspath = prefix + setting + "/checkpoints"
    for exp in os.listdir(takspath):
        if "R500" in exp:
            chkpt_path = os.path.join(takspath, exp, 'Round{}.pt'.format(round_id))
            print(chkpt_path)
            model.load_state_dict(torch.load(chkpt_path, map_location=device))
            model.to(device)
            model.eval()
            all_reprs = {key: list() for key in all_combin_key}
            for batch in test_loader:
                samples = {modal: batch[0][modal].to(device) for modal in all_modal}
                representations_dict = model.get_embedding(samples)
                for key in all_combin_key:
                    all_reprs[key].extend(representations_dict[key].cpu().tolist())
            all_reprs = {key: np.array(all_reprs[key]) for key in all_combin_key}
            R = all_reprs['image+trajectory']
            E = all_reprs['trajectory']
            
            experiment_path = "output/" + setting
            experiment_id = '_'.join([exp.split('_')[7], exp.split('_')[8], exp.split('_')[10]])

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
            dca.cleanup()