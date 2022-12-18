import os
import torch
import numpy as np
from os.path import join
from torchvision import models
from torchvision import transforms
from skimage import io, img_as_ubyte
from tqdm import tqdm
import pandas as pd
from time import time

import gc

from core.CNN_scorers import TorchScorer
from core.insilico_exps import ExperimentEvolution, resize_and_pad_tsr
from core.GAN_utils import upconvGAN
from core.Optimizers import CholeskyCMAES  # HessAware_Gauss_DC,

from mymodel import VisionTransformer

default_init_sigma = 3.0
default_Aupdate_freq = 10


class TransformerScorer(TorchScorer):
    def __init__(self, model_name, imgpix=224, rawlayername=True, device="cuda"):
        self.imgpix = imgpix
        if isinstance(model_name, str):
            if model_name == "vit_b_16":
                self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
                self.layers = list(self.model.encoder.layers) + list(self.model.heads)
                # self.layername = None if rawlayername else layername_dict[model_name]
                self.layername = None
                self.inputsize = (3, imgpix, imgpix)
            elif model_name == "vit_b_32":
                self.model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
                self.layers = list(self.model.encoder.layers) + list(self.model.heads)
                self.layername = None
                self.inputsize = (3, imgpix, imgpix)
            elif model_name == "vit":
                model_kwargs = {
                    "embed_dim": 8,
                    "hidden_dim": 8,
                    "num_heads": 8,
                    "num_layers": 6,
                    "patch_size": 4,
                    "num_channels": 3,
                    "num_patches": 64,
                    "num_classes": 10,
                    "dropout": 0.2,
                }
                model_dic = torch.load('/home/paperspace/mlproj2_new/ActMax-Optimizer-Dev/transformers/model.pt')
                self.model = VisionTransformer(**model_kwargs)
                self.model.load_state_dict(model_dic)
                self.layers = list(self.model.transformer) + list(self.model.mlp_head)
                self.layername = None
                self.inputsize = (3, 32, 32)

        self.model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )  # Imagenet normalization RGB
        self.RGBmean = torch.tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1]).to(device)
        self.RGBstd = torch.tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1]).to(device)
        self.device = device
        self.hooks = []
        self.artiphys = False
        self.record_layers = []
        self.recordings = {}
        self.activation = {}


class TransformerEvolution(ExperimentEvolution):
    def __init__(
        self,
        model_unit,
        max_step=100,
        imgsize=(224, 224),
        corner=(0, 0),
        optimizer=None,
        savedir="",
        explabel="",
        GAN="fc6",
        device="cuda",
    ):
        # super().__init__(
        #     model_unit,
        #     max_step,
        #     imgsize,
        #     corner,
        #     optimizer,
        #     savedir,
        #     explabel,
        #     GAN,
        #     device,
        # )
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        self.pref_unit = model_unit
        # Transformers
        self.model = TransformerScorer(model_unit[0], device=device, imgpix=imgsize[0])
        self.model.select_unit(model_unit)
        # Allow them to choose from multiple optimizers, substitute generator.visualize and render
        if GAN in ["fc6", "fc7", "fc8"]:
            self.G = upconvGAN(name=GAN).cuda()
            self.render_tsr = self.G.visualize_batch_np  # this output tensor
            self.render = self.G.render
            self.code_length = self.G.codelen  # 1000 "fc8" 4096 "fc6", "fc7"
        else:
            raise NotImplementedError
        if optimizer is None:
            self.optimizer = CholeskyCMAES(
                self.code_length,
                population_size=None,
                init_sigma=default_init_sigma,
                init_code=np.zeros([1, self.code_length]),
                Aupdate_freq=default_Aupdate_freq,
                maximize=True,
                random_seed=None,
                optim_params={},
            )
        else:
            self.optimizer = optimizer

        self.max_steps = max_step
        self.corner = corner  # up left corner of the image
        self.imgsize = imgsize  # size of image, allowing showing CNN resized image
        self.savedir = savedir
        self.explabel = explabel
        self.Perturb_vec = []

    def run(self, init_code=None):
        """Same as Resized Evolution experiment"""
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        for self.istep in range(self.max_steps):
            if self.istep == 0:
                if init_code is None:
                    codes = np.zeros([1, self.code_length])
                else:
                    codes = init_code
            t0 = time()
            self.current_images = self.render_tsr(codes)
            t1 = time()  # generate image from code
            # self.current_images = resize_and_pad_tsr(
            #     self.current_images, self.imgsize, self.corner
            # )
            self.current_images = resize_and_pad_tsr(
                self.current_images, self.imgsize, self.corner, canvas_size=self.imgsize
            )
            synscores = self.model.score_tsr(self.current_images)
            t2 = time()  # score images
            codes_new = self.optimizer.step_simple(synscores, codes)
            t3 = time()  # use results to update optimizer
            self.codes_all.append(codes)
            self.scores_all = self.scores_all + list(synscores)
            self.generations = self.generations + [self.istep] * len(synscores)
            codes = codes_new
            # summarize scores & delays
            print(
                "synth img scores: mean {:.3f} +- std {:.3f}".format(
                    np.nanmean(synscores), np.nanstd(synscores)
                )
            )
            print(
                (
                    "step %d  time: total %.2fs | "
                    + "GAN visualize %.2fs   Transformers score %.2fs   optimizer step %.2fs"
                )
                % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2)
            )
        self.codes_all = np.concatenate(tuple(self.codes_all), axis=0)
        self.scores_all = np.array(self.scores_all)
        self.generations = np.array(self.generations)

    def save_best_imgs(self, classname, num):
        idx = np.argmax(self.scores_all)
        select_code = self.codes_all[idx : idx + 1, :]
        score_select = self.scores_all[idx]
        img_select = self.render_tsr(select_code)
        resize_select = resize_and_pad_tsr(img_select, self.imgsize, self.corner)
        resize_select = resize_select.cpu().squeeze().permute((1, 2, 0)).numpy()
        io.imsave(
            join(self.savedir, classname, "Best_%s_%d.png" % (self.explabel, num)),
            img_as_ubyte(resize_select),
        )
        return resize_select


class ImageLoader(object):
    def __init__(self, datapath) -> None:
        self.datapath = datapath
        self._flies_df = pd.DataFrame(columns=["File", "Shape"])
        self._num_imgs = 0
        self.imgs = torch.Tensor([])

    @staticmethod
    def preprocess(img):
        img_tensor = torch.Tensor(img).permute((2, 0, 1)).unsqueeze(dim=0)
        img_new = resize_and_pad_tsr(img_tensor, (224, 224), (0, 0))
        return img_new

    def load(self):
        list_of_files = []
        for root, dirs, files in os.walk(self.datapath):
            for file in files:
                list_of_files.append(os.path.join(root, file))
        for name in tqdm(list_of_files):
            img = io.imread(name)
            shape_tmp = img.shape
            dic_tmp = {"File": [name], "Shape": [str(shape_tmp)]}
            df_tmp = pd.DataFrame(data=dic_tmp)
            self._flies_df = pd.concat([self._flies_df, df_tmp], axis=0)

            img_processed = self.preprocess(img.copy())
            self.imgs = torch.concat([self.imgs, img_processed])
        self._num_imgs = self.imgs.shape[0]
        return self._num_imgs

    def get_image_info(self):
        return self._flies_df


class ExpScores(object):
    def __init__(self, path, model_unit) -> None:
        self._path = path
        self._scores = None
        self.model_unit = model_unit
        # Define the model
        self.model = TransformerScorer(model_name=self.model_unit[0])
        # Select unit
        self.model.select_unit(self.model_unit)

    def load(self):
        loader = ImageLoader(self._path)
        self.num_imgs = loader.load()
        # Check if the imgs are in 255 range
        self._imgs = loader.imgs
        if torch.max(self._imgs) > 1:
            self._imgs = self._imgs / 255.0

    def img_scores(self):
        scores = self.model.score_tsr(self._imgs)
        return scores

    def syn_scores(self, syn=None):
        with torch.no_grad():
            if isinstance(syn, str):
                # Read the syn image
                syn_img = io.imread(syn)
                syn_img_tsr = torch.Tensor(syn_img).permute((2, 0, 1)).unsqueeze(dim=0)
                syn_score = self.model.score_tsr(syn_img_tsr)[0]
            else:
                syn_tsr = syn.permute((2, 0, 1)).unsqueeze(dim=0)
                syn_score = self.model.score_tsr(syn_tsr)[0]
            return syn_score


class ExpSyn(object):
    def __init__(self, model_unit, explabel, optimizer=None, savedir="tmp") -> None:
        self.model_unit = model_unit
        self.optimizer = None
        self.explabel = explabel
        self.model_unit = model_unit
        self.savedir = savedir
        self.img_list = torch.Tensor([])

    def get_syn_images(self, maxiter, classname):
        for i in range(maxiter):
            self.exp = TransformerEvolution(
                self.model_unit,
                max_step=60,
                savedir=self.savedir,
                explabel=self.explabel,
                optimizer=self.optimizer,
            )
            self.exp.run()
            syn_img = self.exp.save_best_imgs(classname, i)
            syn_img = torch.Tensor(syn_img).unsqueeze(dim=0)
            self.img_list = torch.concat([self.img_list, syn_img])
            del self.exp
            gc.collect()
            torch.cuda.empty_cache()
        print("%d Images are generated!" % maxiter)

    def average_syn_image(self):
        return torch.mean(self.img_list, axis=0)
