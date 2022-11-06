# -*- coding: utf-8 -*-
from get_config import get_config
from dataloader.as_dataloader import get_as_dataloader
from get_model import get_model
import os
from utils import validation_constructive
import os
import torch
import wandb
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from losses import laplace_cdf_loss, laplace_cdf
from losses import evidential_loss, evidential_prob_vacuity
from losses import SupConLoss
from visualization.vis import plot_tsne_visualization
import dataloader.utils as utils
from utils import validation_constructive
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from DDU.utils.temperature_scaling import ModelWithTemperature
from DDU.metrics.calibration_metrics import expected_calibration_error
from DDU.metrics.uncertainty_confidence import entropy, logsumexp
#from DDU.metrics.ood_metrics import get_roc_auc, get_roc_auc_logits, get_roc_auc_ensemble

# Import GMM utils
from DDU.utils.gmm_utils import get_embeddings, gmm_evaluate, gmm_fit


class TransformerFeatureMap:
    def __init__(self, model, layer_name='avgpool'):
        self.model = model
        for name, module in self.model.named_modules():
            if layer_name in name:
                module.register_forward_hook(self.get_feature)
        self.feature = []

    def get_feature(self, module, input, output):
        self.feature.append(output.cpu())

    def __call__(self, input_tensor):
        self.feature = []
        with torch.no_grad():
            output = self.model(input_tensor.cuda())

        return self.feature
    
    

class Network(object):
    """Wrapper for training and testing pipelines."""

    def __init__(self, model, config):
        """Initialize configuration."""
        self.config = config
        self.model = None
        self.model = model
        self.encoder = TransformerFeatureMap(self.model.cuda())
        if self.config['use_cuda']:  
            print(torch.cuda.device_count())
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        self.num_classes_AS = config['num_classes']
        if config['cotrastive_method']=='Linear':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config['lr'])
            
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,T_max=config['num_epochs'])
        self.loss_type = config['loss_type']
        self.contrastive_method = config['cotrastive_method']
        self.temperature = config['temp']
        #self.bicuspid_weight = config['bicuspid_weight']
        # Recording the training losses and validation performance.
        self.train_losses = []
        self.valid_oas = []
        self.idx_steps = []

        # init auxiliary stuff such as log_func
        self._init_aux()

    def _init_aux(self):
        """Intialize aux functions, features."""
        # Define func for logging.
        self.log_func = print

        # Define directory wghere we save states such as trained model.
        if self.config['use_wandb']:
            if self.config['mode'] == 'test':
                # read from the pre-specified test folder
                if self.config['model_load_dir'] is None:
                    raise AttributeError('For test-only mode, please specify the model state_dict folder')
                self.log_dir = os.path.join(self.config['log_dir'], self.config['model_load_dir'])
            else:
                # create a new directory
                self.log_dir = os.path.join(self.config['log_dir'], wandb.run.name)
        else:
            self.log_dir = self.config['log_dir']
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # File that we save trained model into.
        self.checkpts_file = os.path.join(self.log_dir, "checkpoint.pth")

        # We save model that achieves the best performance: early stopping strategy.
        self.bestmodel_file = os.path.join(self.log_dir, "best_model.pth")
        #self.bestmodel_file = Path('/AS_clean/AS_thesis/fixmatch_as_0_2/model_best.pth')
        self.bestmodel_file_contrastive = os.path.join(self.log_dir, "best_model_cont.pth")
        
        # For test, we also save the results dataframe
        self.test_results_file = os.path.join(self.log_dir, "best_model.pth")
 
    
    def _save(self, pt_file):
        """Saving trained model."""

        # Save the trained model.
        torch.save(
            {
                #"model": self.model.module.state_dict(),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            pt_file,
        )

    def _restore_module(self, pt_file):
        """Restoring trained model."""
        print(f"restoring {pt_file}")
        
        # original saved file with DataParallel
        state_dict = torch.load(pt_file)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict["model"].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        # Loading model.
        self.model.load_state_dict(new_state_dict)
        # Loading optimizer.
        self.optimizer.load_state_dict(state_dict["optimizer"])
        
    def _restore(self, pt_file):
        """Restoring trained model."""
        print(f"restoring {pt_file}")

        # Read checkpoint file.
        load_res = torch.load(pt_file)
        # Loading model.
        self.model.load_state_dict(load_res["model"])
        # Loading optimizer.
        self.optimizer.load_state_dict(load_res["optimizer"])
        
        
        
    
    def _get_loss(self, logits, target, nclasses):
        """ Compute loss function based on configs """
        if self.loss_type == 'cross_entropy':
            # BxC logits into Bx1 ground truth
            loss = F.cross_entropy(logits, target)
        elif self.loss_type == 'evidential':
            # alpha = F.softplus(logits)+1
            # target_oh = F.one_hot(target, nclasses)
            # mse, kl = evidential_mse(alpha, target_oh, alpha.device)
            # loss = torch.mean(mse + 0.1*kl)
            loss = evidential_loss(logits, target, nclasses)
        elif self.loss_type == 'laplace_cdf':
            # logits_categorical = laplace_cdf(F.sigmoid(logits), nclasses, logits.device)
            # target_oh = 0.9*F.one_hot(target, nclasses) + 0.1/nclasses
            # loss = F.binary_cross_entropy(logits_categorical, target_oh)
            loss = laplace_cdf_loss(logits, target, nclasses)
        elif self.loss_type == 'SupCon' or self.loss_type == 'SimCLR':
            criterion = SupConLoss(temperature=self.temperature)
            if torch.cuda.is_available():
                criterion = criterion.cuda()
            if self.contrastive_method == 'SupCon':
                loss = criterion(logits, target)
            elif self.contrastive_method == 'SimCLR':
                loss = criterion(logits)
        else:
            raise NotImplementedError
        return loss
    
    # obtain summary statistics of
    # argmax, max_percentage, entropy, evid.uncertainty for each function
    # expects logits BxC for classification, Bx2 for cdf
    def _get_prediction_stats(self, logits, nclasses):
        # convert logits to probabilities
        if self.loss_type == 'cross_entropy':
            prob = F.softmax(logits, dim=1)
            vacuity = -1
        elif self.loss_type == 'evidential':
            prob, vacuity = evidential_prob_vacuity(logits, nclasses)
            #vacuity = vacuity.squeeze()
        elif self.loss_type == 'laplace_cdf':
            prob = laplace_cdf(F.sigmoid(logits), nclasses, logits.device)
            vacuity = -1
        else:
            raise NotImplementedError
        max_percentage, argm = torch.max(prob, dim=1)
        entropy = torch.sum(-prob*torch.log(prob), dim=1)
        uni = utils.test_unimodality(prob.cpu().numpy())
        return argm, max_percentage, entropy, vacuity, uni
        
    # def get_lr(self):
    #     return self.scheduler.get_lr()
    
    def model_initialization(self,path):
        self._restore_module(path)
        # Switch the model into eval mode.
        self.model.eval()
        

    @torch.no_grad()
    def info_extractor(self,loader, model=None , mode="test",record_embeddings=False,info = None):
        """Logs the network outputs in dataloader
        computes per-patient preds and outputs result to a DataFrame"""
        print('NOTE: test_comprehensive mode uses batch_size=1 to correctly display metadata')
        fn, patient, view, age, lv, as_label,bicuspid = [], [], [], [], [], [],[]
        target_AS_arr, target_B_arr, pred_AS_arr, pred_B_arr = [], [], [], []
        max_AS_arr, entropy_AS_arr, vacuity_AS_arr, uni_AS_arr = [], [], [], []
        #max_B_arr, entropy_B_arr, vacuity_B_arr = [], [], []
        predicted_qual = []
        embeddings = []
        
        for cine, target_AS, target_B, data_info, cine_orig in tqdm(loader):
            # collect the label info
            target_AS_arr.append(int(target_AS[0]))
            target_B_arr.append(int(target_B[0]))
            # Transfer data from CPU to GPU.
            if self.config['use_cuda']:
                cine = cine.cuda()
                target_AS = target_AS.cuda()
                target_B = target_B.cuda()
                
            # collect metadata from data_info
            fn.append(data_info['path'][0])
            patient.append(int(data_info['patient_id'][0]))
            view.append(data_info['view'][0])
            age.append(int(data_info['age'][0]))
            #lv.append(float(data_info['LVMass indexed'][0]))
            as_label.append(data_info['as_label'][0])
            bicuspid.append(data_info['Bicuspid'][0])
            # pvq = (data_info['predicted_view_quality'][0] * 
            #        data_info['predicted_view_probability'][0]).cpu().numpy()
            # predicted_qual.append(pvq)
            
            # get the model prediction
            # pred_AS, pred_B = self.model(cine) #1x3xTxHxW
            if model == None:
                pred_AS= self.model(cine) #1x3xTxHxW
            else:
                pred_AS= model(cine) #1x3xTxHxW
            self.encoder = TransformerFeatureMap(self.model)
            embedding = self.encoder(cine)    
            # collect the model prediction info
            argm, max_p, ent, vac, uni = self._get_prediction_stats(pred_AS, self.num_classes_AS)
            pred_AS_arr.append(argm.cpu().numpy()[0])
            max_AS_arr.append(max_p.cpu().numpy()[0])
            entropy_AS_arr.append(ent.cpu().numpy()[0])
            if self.loss_type == 'evidential':
                vacuity_AS_arr.append(vac.cpu().numpy()[0])
            else:
                vacuity_AS_arr.append(vac)
            uni_AS_arr.append(uni[0])
            
            if record_embeddings:
                embeddings += [embedding[0].squeeze().numpy()]

                
        # compile the information into a dictionary
        d = {'path':fn, 'id':patient, 'view':view, 'age':age, 'as':as_label, 'bicuspid': bicuspid ,
             'GT_AS':target_AS_arr, 'pred_AS':pred_AS_arr, 'max_AS':max_AS_arr,
             'ent_AS':entropy_AS_arr, 'vac_AS':vacuity_AS_arr, 'uni_AS':uni_AS_arr,
             # 'GT_B':target_B_arr, 'pred_B':pred_B_arr, 'max_B':max_B_arr,
             # 'ent_B':entropy_B_arr, 'vac_B':vacuity_B_arr, 
             }
        df = pd.DataFrame(data=d)
        cf = confusion_matrix(target_AS_arr, pred_AS_arr)
        accuracy =  accuracy_score(target_AS_arr,pred_AS_arr)
        if info == None:
            #save the dataframe
            test_results_file = os.path.join(self.log_dir, mode+".csv")
            df.to_csv(test_results_file)
            if record_embeddings:
                embeddings = np.array(embeddings)
                #num_batches, b, d = embeddings.shape
                #print(num_batches, b, d)
                #embeddings = np.reshape(embeddings, (num_batches*b, d))
                #tsne_save_file =  os.path.join(self.log_dir, mode+"_tsne.html")
                #plot_tsne_visualization(X=embeddings, y=as_label, info=fn, title=tsne_save_file , b = bicuspid)
        if info == 'classification':
            return ( cf , accuracy , target_AS_arr, pred_AS_arr , max_AS_arr,)
        elif info == 'embedding':
            return (embeddings , target_AS_arr)
        
    def temperature_scling(self, loader_va,loader_te):
        print('getting the temperature:')
        temp_scaled_net = ModelWithTemperature(self.model)
        temp_scaled_net.set_temperature(loader_va)
        topt = temp_scaled_net.temperature
        return net.info_extractor(loader = dataloader_te, model = temp_scaled_net,mode="test",info = 'classification' ,record_embeddings=False)
    
    def gmm_forward(self, gaussians_model , data_B_X):
        self.encoder = TransformerFeatureMap(self.model)
        features_B_Z = self.encoder(data_B_X)
        log_probs_B_Y = gaussians_model.log_prob(torch.tensor(features_B_Z[0].squeeze()))
        return log_probs_B_Y


    def gmm_evaluate(self, gaussians_model , loader):
        
        num_samples = len(loader.dataset)
        logits_N_C = torch.empty((num_samples, 4), dtype=torch.float)
        labels_N = torch.empty(num_samples, dtype=torch.int)
        pred_N = torch.empty(num_samples, dtype=torch.float)
        entropy_N = torch.empty(num_samples, dtype=torch.int)
        with torch.no_grad():
            start = 0
            for data, label,_,_,_ in tqdm(loader):
                data = data.cuda()
                label = label.cuda()
                pred_AS= self.model(data)
                argm, max_p, ent, vac, uni = self._get_prediction_stats(pred_AS,
                                                                        self.num_classes_AS)
                
                logit_B_C = net.gmm_forward(gaussians_model, data)

                end = start + len(data)
                logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
                labels_N[start:end].copy_(label, non_blocking=True)
                pred_N[start:end].copy_(argm, non_blocking=True)
                entropy_N[start:end].copy_(ent, non_blocking=True)
                start = end

        return logits_N_C, labels_N, pred_N, entropy_N
    


    def get_roc_auc_logits(self,logits, ood_logits, uncertainty, confidence=False):
        uncertainties = uncertainty(logits)
        #ood_uncertainties = uncertainty(ood_logits)

        # In-distribution
        bin_labels = torch.zeros(uncertainties.shape[0]).to(device)
        in_scores = uncertainties

        # OOD
        #bin_labels = torch.cat((bin_labels, torch.ones(ood_uncertainties.shape[0]).to(device)))

        if confidence:
            bin_labels = 1 - bin_labels
        ood_scores = ood_uncertainties  # entropy(ood_logits)
        scores = torch.cat((in_scores, ood_scores))

        fpr, tpr, thresholds = metrics.roc_curve(bin_labels.cpu().numpy(), scores.cpu().numpy())
        precision, recall, prc_thresholds = metrics.precision_recall_curve(bin_labels.cpu().numpy(), scores.cpu().numpy())
        auroc = metrics.roc_auc_score(bin_labels.cpu().numpy(), scores.cpu().numpy())
        auprc = metrics.average_precision_score(bin_labels.cpu().numpy(), scores.cpu().numpy())

        return (fpr, tpr, thresholds), (precision, recall, prc_thresholds), auroc, auprc



if __name__ == "__main__":
    
    config = get_config()

    model = get_model(config)
    net = Network(model, config)
    net.model_initialization(Path('/AS_clean/AS_thesis/bestlogs/all_data_dgx/best_model.pth'))
    
    dataloader_tr = get_as_dataloader(config, split='train', mode='test')
    dataloader_va = get_as_dataloader(config, split='val', mode='test')
    dataloader_te = get_as_dataloader(config, split='test', mode='test')
    
    
    # ECE calculation
    # (conf_matrix, accuracy, labels_list, predictions, confidences,) = net.info_extractor(dataloader_te, mode="test",info = 'classification' ,record_embeddings=False)
    # ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)
    
    # t_ECE calculation
    # (t_conf_matrix, t_accuracy, t_labels_list, t_predictions, t_confidences,) = net.temperature_scling(dataloader_va,dataloader_te)
    # t_ece = expected_calibration_error(t_confidences, t_predictions, t_labels_list, num_bins=15)
    # print('ece',ece,'t_ece',t_ece)
    
    # Building the GMM model
    print("GMM Model, Getting train embeddings")
    embeddings, labels = net.info_extractor(dataloader_va, mode="test",info = 'embedding' ,record_embeddings=True)
    
    # GMM
    gaussians_model, jitter_eps = gmm_fit(embeddings=torch.tensor(embeddings),
                                          labels=torch.tensor(labels), num_classes=4)
    logits_te, labels_te, pred_te, entr_te = net.gmm_evaluate(gaussians_model,dataloader_te)
    logits_val, labels_val, pred_val, entr_val = net.gmm_evaluate(gaussians_model,dataloader_va)
    
    # Uncertainty - Density - val
    val_densities = torch.logsumexp(logits_val, dim=-1)
    val_min_density = val_densities.min().item()
    val_median_density = torch.mean(val_densities).item()
    
    # Uncertainty - Density - test
    te_densities = torch.logsumexp(logits_te, dim=-1)
    te_min_density = te_densities.min().item()
    te_median_density = torch.mean(te_densities).item()
    
    #
    d = {'Label':labels_val,'Prediction':pred_val, 
         'Entropy':entr_val, 'Density':val_densities - val_min_density,
         }
    df = pd.DataFrame(data=d)
    df.to_csv('validation_uncertainty')

    
    d = {'Label':labels_te,'Prediction':pred_te, 
     'Entropy':entr_te, 'Density':te_densities - te_min_density,
     }
    df = pd.DataFrame(data=d)
    df.to_csv('validation_uncertainty')