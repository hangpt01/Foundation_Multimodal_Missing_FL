import utils.logger.basic_logger as bl
import numpy as np
import utils.system_simulator as ss

class Logger(bl.Logger):
    def initialize(self):
        """This method is used to record the stastic variables that won't change across rounds (e.g. local data size)"""
        for c in self.clients:
            self.output['client_datavol'].append(len(c.train_data))

    """This logger only records metrics on validation dataset""" 
    # This is for valid set in each client - not calling
    def log_once(self, *args, **kwargs):
        self.info('Current_time:{}'.format(ss.clock.current_time))
        valid_metrics = self.server.test_on_clients('valid')
        for met_name, met_val in valid_metrics.items():
            self.output['valid_'+met_name+'_dist'].append(met_val)
            self.output['valid_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(self.server.local_data_vols, met_val)]) / self.server.total_data_vol)
            self.output['mean_valid_' + met_name].append(np.mean(met_val))
            self.output['std_valid_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()
        
    # This is for the global valid set    
    # def log_once(self, *args, **kwargs):
    #     self.info('Current_time:{}'.format(ss.clock.current_time))
    #     valid_metrics = self.server.test_on_clients()
    #     valid_metrics = self.server.client()
    #     self.output['valid_loss'].append(valid_metrics)
    #     # import pdb; pdb.set_trace()
    #     # for met_name, met_val in valid_metrics.items():
    #     #     self.output['valid_'+met_name+'_dist'].append(met_val)
    #     #     self.output['valid_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(self.server.local_data_vols, met_val)]) / self.server.total_data_vol)
    #     #     self.output['mean_valid_' + met_name].append(np.mean(met_val))
    #     #     self.output['std_valid_' + met_name].append(np.std(met_val))
    #     # output to stdout
    #     self.show_current_output()