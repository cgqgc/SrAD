import copy
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from sklearn import metrics
from sklearn import metrics as mts
from sklearn.manifold import TSNE
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from thop import profile
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.data_utils import get_data_path, get_transform
from dataloaders.dataload import BraTSAD, PseADSeg
from networks.unet import UNet
from utils.util import AELoss, AverageMeter, compute_best_dice

os.environ["WANDB_API_KEY"] = 'a12352aa3e0666b79cf5061e9b17d952e799b7ac'
os.environ["WANDB_MODE"] = "offline"

class DAEWorker:
    def __init__(self, opt):
        super(DAEWorker, self).__init__()
        self.project_name = None
        self.logger = None
        self.opt = opt
        self.seed = None
        self.seed = None
        self.train_set = None
        self.test_set = None
        self.train_loader = None
        self.test_loader = None
        self.scheduler = None

        self.net = UNet(in_channels=self.opt.model['in_c'], n_classes=self.opt.model['in_c']).cuda()
        self.criterion = AELoss()

        self.optimizer = torch.optim.Adam(self.net.parameters(), self.opt.train['lr'],
                                            weight_decay=self.opt.train['weight_decay'])
        
        self.pixel_metric = True 
        self.noise_res = 16
        self.noise_std = 0.2

    def set_gpu_device(self):
        torch.cuda.set_device(self.opt.gpu)
        print("=> Set GPU device: {}".format(self.opt.gpu))

    def set_seed(self):
        self.seed = self.opt.train['seed']
        if self.seed is None:
            self.seed = np.random.randint(1, 999999)
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        seed_path = os.path.join(self.opt.train['save_dir'], 'seed.txt')
        with open(seed_path, "w") as f:
            f.write(str(self.seed))

    def set_dataloader(self, test=False):
        data_path = get_data_path(dataset=self.opt.dataset)
        train_transform = get_transform(self.opt)
        test_transform = get_transform(self.opt)

        if self.opt.dataset == 'brats':
            if not test:
                self.train_set = BraTSAD(main_path=data_path, img_size=self.opt.model['input_size'],
                                         transform=train_transform, mode='train')
            self.test_set = BraTSAD(main_path=data_path, img_size=self.opt.model['input_size'],
                                    transform=test_transform, mode='test')
        elif self.opt.dataset == 'pseseg':
            if not test:
                self.train_set = PseADSeg(main_path=data_path, img_size=self.opt.model['input_size'],
                                         transform=train_transform, mode='train')
            self.test_set = PseADSeg(main_path=data_path, img_size=self.opt.model['input_size'],
                                    transform=test_transform, mode='test')
        else:
            raise Exception("Invalid dataset: {}".format(self.opt.dataset))

        if not test:
            self.train_loader = DataLoader(self.train_set, batch_size=self.opt.train['batch_size'], shuffle=True,pin_memory=True,num_workers=0)
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False,pin_memory=True,num_workers=0)

    def set_logging(self, test=False):
        example_in = torch.zeros((1, self.opt.model["in_c"],
                                  self.opt.model['input_size'], self.opt.model['input_size'])).cuda()

        # flops, params = profile(self.net, inputs=(example_in,))
        flops, params = profile(copy.deepcopy(self.net), inputs=(example_in,))

        flops, params = round(flops * 1e-6, 4), round(params * 1e-6, 4)  # 1e6 = M
        flops, params = str(flops) + "M", str(params) + "M"

        exp_configs = {"dataset": self.opt.dataset,
                       "model": self.opt.model["name"],
                       "in_channels": self.opt.model["in_c"],
                       "input_size": self.opt.model['input_size'],

                       "epochs": self.opt.train["epochs"],
                       "batch_size": self.opt.train["batch_size"],
                       "lr": self.opt.train["lr"],
                       "weight_decay": self.opt.train["weight_decay"],
                       "seed": self.seed,

                       "num_params": params,
                       "FLOPs": flops}

        if not test:
            self.logger = wandb.init(project=self.opt.project_name, config=exp_configs)
        print("============= Configurations =============")
        for key, values in exp_configs.items():
            print(key + ":" + str(values))
        print()

    def close_network_grad(self):
        for param in self.net.parameters():
            param.requires_grad = False


    def enable_network_grad(self):
        for param in self.net.parameters():
            param.requires_grad = True


    def set_test_loader(self):
        """ Use for only evaluation"""
        data_path = get_data_path(dataset=self.opt.dataset)
        test_transform = get_transform(self.opt)
        
        if self.opt.dataset == 'brats':
            self.test_set = BraTSAD(main_path=data_path, img_size=self.opt.model['input_size'],
                                    transform=test_transform, mode='test')
        elif self.opt.dataset == 'pseseg':
            self.test_set = PseADSeg(main_path=data_path, img_size=self.opt.model['input_size'],
                                    transform=test_transform, mode='test')
        else:
            raise Exception("Invalid dataset: {}".format(self.opt.dataset))
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False,pin_memory=True,num_workers=0)
        print("=> Set test dataset: {} | Input size: {} | Batch size: {}".format(self.opt.dataset,
                                                                                 self.opt.model['input_size'], 1))

    def save_checkpoint(self,file_name=None):
        if file_name is not None:
            torch.save(self.net.state_dict(), os.path.join(self.opt.train['save_dir'], "checkpoints", file_name))
        else:
            torch.save(self.net.state_dict(), os.path.join(self.opt.train['save_dir'], "checkpoints", "model.pt"))


    def load_checkpoint(self, file_name=None):
        if file_name is not None:
            model_path = os.path.join(self.opt.train['save_dir'], "checkpoints", "model.pt")
            self.net.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:{}".format(self.opt.gpu))))
            print("=> Load model from {}".format(model_path))
        else:
            model_path = os.path.join(self.opt.train['save_dir'], "checkpoints", "model_last.pt")
            self.net.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:{}".format(self.opt.gpu))))
            print("=> Load model from {}".format(model_path))

    def run_eval(self):
        results,epoch_auc,test_labels,test_scores,test_names = self.evaluate()
        metrics_save_path = os.path.join(self.opt.train['save_dir'], "metrics.txt")
        with open(metrics_save_path, "w") as f:
            for key, value in results.items():
                f.write(str(key) + ": " + str(value) + "\n")
                print(key + ": {:.4f}".format(value))
        with open(f"{self.opt.test['save_dir']}/{self.opt.model['name']}_{self.opt.dataset}_score.csv", "w") as f:
            f.write("patient_id,label,img_distance\n")
            for batch in zip(test_names, test_labels, test_scores):
                name = batch[0]
                label = batch[1]
                score = batch[2]
                f.write(f"{name},{label},{score}\n")
        self.save_score()
    
    def save_score(self):
        df = pd.read_csv(f"{self.opt.test['save_dir']}/{self.opt.model['name']}_{self.opt.dataset}_score.csv")
        test_dir = os.path.join(self.opt.test['save_dir'],'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        trainig_label = 0
        labels =  df["label"]  # np.where(df["label"].values == trainig_label, 0, 1)
        img_distance = df["img_distance"].values
        c_img_distance = img_distance
        
        fpr, tpr, threshold = roc_curve(labels, img_distance)
        precision, recall, _ = precision_recall_curve(labels, img_distance)
        roc_auc = auc(fpr, tpr)
        pr_auc =  auc(recall, precision)

        youden = tpr-fpr
        best_idx = np.argmax(youden)
        best_threshold = threshold[best_idx]
        img_distance[img_distance>best_threshold] = 1
        img_distance[img_distance<=best_threshold] = 0
        C = mts.confusion_matrix(labels, img_distance)
        plt.matshow(C, cmap=plt.cm.Blues) # 根据最下面的图按自己需求更改颜色
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
        
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.legend()
        plt.savefig(f"{test_dir}/confusion matrix.png") 
        plt.close()

        tn, fp, fn, tp = C.ravel()
        f1_score = mts.f1_score(labels, img_distance, average='macro')
        specificity = tn / (tn + fp)
        recall_score = tp / (tp + fn) #sensitivity
        precision_score = tp / (tp + fp)
        acc = (tp + tn) / (tp + tn + fp + fn)

        metrics = {
            'AUC': round(roc_auc,3),
            'Accuracy': round(acc,3),
            'F1-Score': round(f1_score,3),
            'Recall': round(recall_score,3),
            'Specificity': round(specificity,3),
            'Precision': round(precision_score,3)
        }

        df = pd.DataFrame(metrics,index=[0])
        df.to_csv(f"{test_dir}/metrics.csv",index=False)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {round(roc_auc,3)}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC-AUC")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(f"{test_dir}/roc.png") 
        plt.close()

        plt.figure()
        plt.plot(recall, precision, label=f"PR = {round(pr_auc,3)}")
        plt.title("PR-AUC")
        plt.xlabel("Recall")
        plt.ylabel("Pecision")
        plt.legend()
        plt.savefig(f"{test_dir}/pr.png") 
        plt.close()

    def train_epoch(self):
        self.net.train()
        losses = AverageMeter()
        for idx_batch, data_batch in enumerate(self.train_loader):
            img = data_batch['img']
            img = img.cuda()
            noisy_img, noise_tensor = self.add_noise(img)

            net_out = self.net(noisy_img)

            loss = self.criterion(img, net_out)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), img.size(0))
        return losses.avg
    def add_noise(self, x):
        """
        Generate and apply randomly translated noise to batch x
        """

        # input N x C x H x W
        # to apply it in for rgb maybe not take diff noise for each channel? (input.shape[1] should be 1)
        ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], self.noise_res, self.noise_res),
                          std=self.noise_std).cuda()

        ns = F.interpolate(ns, size=self.opt.model['input_size'], mode='bilinear', align_corners=True)

        # Roll to randomly translate the generated noise.
        roll_x = random.choice(range(self.opt.model['input_size']))
        roll_y = random.choice(range(self.opt.model['input_size']))
        ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

        # Use foreground mask for MRI, to only apply noise in the foreground.
        if self.opt.dataset in ['brain', 'brats']:
            mask = (x > x.min())
            ns *= mask
        # if config.center:
        ns = (ns - 0.5) * 2
        res = x + ns

        return res, ns

    @torch.no_grad()
    def evaluate(self):
        self.net.eval()
        self.close_network_grad()

        test_imgs, test_imgs_hat, test_scores, test_score_maps, test_names, test_labels, test_masks = \
            [], [], [], [], [], [], []
        refine_anomaly_maps = []
        abnormal_anomaly_maps = []
        abnormal_masks = []

        old_dices = 0.0
        new_dices = 0.0
        old_recalls = 0.0
        new_recalls = 0.0
        old_specs = 0.0
        new_specs = 0.0

        # with torch.no_grad():
        for idx_batch, data_batch in tqdm(enumerate(self.test_loader)):
            # test batch_size=1
            img, label, name = data_batch['img'], data_batch['label'], data_batch['name']
                
            img = img.cuda()
            mask = data_batch['mask']
            net_out = self.net(img)
            # net_out = torch.nn.functional.tanh(net_out)

            anomaly_score_map = self.criterion(img, net_out, anomaly_score=True, keepdim=True).detach().cpu()
            #原
            test_score_maps.append(anomaly_score_map)
                
            test_names.append(name)
            test_labels.append(label.item())
            if self.pixel_metric:
                test_masks.append(mask)

            if self.opt.test['save_flag']:
                img_hat = net_out['x_hat']
                test_names.append(name)
                test_imgs.append(img.cpu())
                test_imgs_hat.append(img_hat.cpu())
                # z = net_out['z']
                # test_repts.append(z.cpu().detach().numpy())

        test_score_maps = torch.cat(test_score_maps, dim=0)  # Nx1xHxW


        test_scores = torch.mean(test_score_maps, dim=[1, 2, 3]).numpy()  # N

        # image-level metrics
        test_labels = np.array(test_labels)
        auc = metrics.roc_auc_score(test_labels, test_scores)
        ap = metrics.average_precision_score(test_labels, test_scores)

        results = {'AUC': auc, "AP": ap}

        # pixel-level metrics
        if self.pixel_metric:
            test_masks = torch.cat(test_masks, dim=0).unsqueeze(1)  # NxHxW -> Nx1xHxW
            # abnormal_masks = torch.cat(abnormal_masks, dim=0).unsqueeze(1)  # NxHxW -> Nx1xHxW
            pix_ap = metrics.average_precision_score(test_masks.numpy().reshape(-1),
                                                     test_score_maps.numpy().reshape(-1))
            pix_auc = metrics.roc_auc_score(test_masks.numpy().reshape(-1),
                                            test_score_maps.numpy().reshape(-1))

            best_dice, best_thresh = compute_best_dice(test_score_maps.numpy(), test_masks.numpy())
            

            results.update({'PixAUC': pix_auc, 'PixAP': pix_ap, 
                            'BestDice': best_dice, 'BestThresh': best_thresh})
            test_masks = None

        # others
        test_normal_score = np.mean(test_scores[np.where(test_labels == 0)])
        test_abnormal_score = np.mean(test_scores[np.where(test_labels == 1)])
        results.update({"normal_score": test_normal_score, "abnormal_score": test_abnormal_score})
        
        self.enable_network_grad()
        return results, auc, test_labels, test_scores,test_names
    

    def run_train(self):
        num_epochs = self.opt.train['epochs']
        print("=> Initial learning rate: {:g}".format(self.opt.train['lr']))
        t0 = time.time()
        max_auc = 0.0
        max_dice = 0.0
        max_abnormal_only_dice = 0.0
        max_lesion_only_dice = 0.0
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            self.logger.log(step=epoch, data={"train/loss": train_loss})
            # self.logger.log(step=epoch, data={"train/loss": train_loss, "train/lr": self.scheduler.get_last_lr()[0]})
            # self.scheduler.step()

            if epoch == 1 or epoch % self.opt.train['eval_freq'] == 0:
                eval_results, epoch_auc,test_labels,test_scores,test_names = self.evaluate()

                t = time.time() - t0
                print("Epoch[{:3d}/{:3d}]  Time:{:.1f}s  loss:{:.5f}".format(epoch, num_epochs, t, train_loss),
                      end="  |  ")

                keys = list(eval_results.keys())
                for key in keys:
                    print(key+": {:.5f}".format(eval_results[key]), end="  ")
                    eval_results["val/"+key] = eval_results.pop(key)
                print()
                epoch_dice = eval_results['val/BestDice']
                # abnormal_epoch_dice = eval_results['val/abnormal_only_dice']
                # lesion_only_epoch_dice = eval_results['val/lesion_only_dice']

                self.logger.log(step=epoch, data=eval_results)
                t0 = time.time()

                if epoch_dice > max_dice:
                    max_dice = epoch_dice
                    self.save_checkpoint("bmodel_dice.pt")


                if epoch_auc > max_auc:
                    max_auc = epoch_auc

                    fpr, tpr, threshold = metrics.roc_curve(test_labels, test_scores)
                    precision, recall, _ = metrics.precision_recall_curve(test_labels, test_scores)
                    roc_auc = metrics.auc(fpr, tpr)
                    pr_auc =  metrics.auc(recall, precision)

                    best_thresh = threshold[np.argmax(tpr - fpr)]
                    test_scores[test_scores>best_thresh] = 1
                    test_scores[test_scores<=best_thresh] = 0
                    C = metrics.confusion_matrix(test_labels, test_scores)

                    plt.matshow(C, cmap=plt.cm.Blues) # 根据最下面的图按自己需求更改颜色
                    for i in range(len(C)):
                        for j in range(len(C)):
                            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
                    
                    plt.title('Confusion Matrix')
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')
                    plt.legend()
                    plt.savefig(os.path.join(self.opt.test['save_dir'] , "confusion matrix.png")) 
                    plt.close()

                    tn, fp, fn, tp = C.ravel()
                    f1_score = metrics.f1_score(test_labels, test_scores, average='macro')
                    specificity = tn / (tn + fp)
                    recall_score = tp / (tp + fn) #sensitivity
                    precision_score = tp / (tp + fp)
                    acc = (tp + tn) / (tp + tn + fp + fn)

                    mts = {
                        'AUC': round(roc_auc,3),
                        'Accuracy': round(acc,3),
                        'F1-Score': round(f1_score,3),
                        'Recall': round(recall_score,3),
                        'Specificity': round(specificity,3),
                        'Precision': round(precision_score,3),
                        'Dice': round(epoch_dice,3),
                        'BestDice': round(max_dice,3),
                        'seed': self.seed
                    }

                    df = pd.DataFrame(mts,index=[0])
                    df.to_csv(os.path.join(self.opt.test['save_dir'] , "cls_metrics.csv"),index=False)

                    plt.figure()
                    plt.plot(fpr, tpr, label=f"AUC = {round(roc_auc,3)}")
                    plt.plot([0, 1], [0, 1], linestyle="--")
                    plt.title("ROC-AUC")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.legend()
                    plt.savefig(os.path.join(self.opt.test['save_dir'], "roc.png"))
                    plt.close()

                    plt.figure()
                    plt.plot(recall, precision, label=f"PR = {round(pr_auc,3)}")
                    plt.title("PR-AUC")
                    plt.xlabel("Recall")
                    plt.ylabel("Pecision")
                    plt.legend()
                    plt.savefig(os.path.join(self.opt.test['save_dir'], "pr.png"))
                    plt.close()

                    self.save_checkpoint("bmodel_auc.pt")


        self.save_checkpoint(file_name='model_last.pt')
        self.logger.finish()
