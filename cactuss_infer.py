#!/usr/bin/env python
import os
from glob import glob
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor
from PIL import Image as pil


sys.path.append('cut')
# from cut.options.base_options import BaseOptions
from cut.options.test_options import TestOptions
from cut.models import create_model
from cut.data import create_dataset
from cut.data.base_dataset import get_transform

# CUT_CKPT_DIR = './ckpt'
# CUT_CKPT_DIR = '/home/steven_kuang/Documents/GR/cactuss/cut/checkpoints/transverse_aorta'
CUT_CKPT_DIR = '/media/steven_kuang/My Passport/Work/Cactuss_models/transverse_aorta_230_nocath'
DATAROOT_CUT = './cut/datasets/aorta_for_val'

try:
    import gdown
except ModuleNotFoundError:
    raise AssertionError('This example requires `gdown` to be installed. '
                         'Please install using `pip install gdown`')

DATASET_FOLDER = './cut/datasets/test_imgs'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor2numpy(tensor_img):
    numpy_img = tensor_img.data[0].clamp(-1.0, 1.0).cpu().float().numpy()
    numpy_img = (np.transpose(numpy_img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return numpy_img.astype(np.uint8)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class CACTUSS:
    def __init__(self):
        # test parameters for cut network
        testoptions = TestOptions()
        self.opt = testoptions.gather_options()
        self.opt.dataroot = DATASET_FOLDER
        self.opt.checkpoints_dir = CUT_CKPT_DIR
        self.opt.gpu_ids = ''
        self.opt.num_threads = 0
        self.opt.batch_size = 1  # test code only supports batch_size = 1
        self.opt.serial_batches = True  # disable data shuffling
        self.opt.no_flip = True
        self.opt.display_id = -1
        self.opt.name = ''
        self.opt.isTrain = False
        testoptions.print_options(self.opt)
        #self.cut_model = self.load_cut_net()
        self.cut_model, self.cut_opt = self.load_cut()

        self.a_paths = DATAROOT_CUT + '/testA/p1_sweep2_8.png'
        self.b_paths = DATAROOT_CUT + '/testB/CT007_68.png'
        self.real_B = cv2.imread(self.b_paths)
        self.real_B = pil.fromarray(self.real_B)

    def load_cut(self):
        opt = TestOptions().parse()  # get test options
        opt.dataroot = DATAROOT_CUT

        # hard-code some parameters for test
        # opt.gpu_ids = '1'
        opt.num_threads = 0  # test code only supports num_threads = 1
        opt.batch_size = 1  # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
        opt.name = 'aorta_CUT'

        opt.checkpoints_dir = self.opt.checkpoints_dir #+ '/aorta_CUT' #+ opt.name
        files = sorted(glob(opt.checkpoints_dir + '/' + opt.name + '/*.pth'))
        print("files: ", opt.checkpoints_dir)

        opt.epoch = files[0].split("/")[-1].split("_")[0]
        print("EPOCH: ", opt.epoch)

        # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
        model = create_model(opt)  # create a model given opt.model and other options
        print('CUT Model device: ', model.device)
        # model=model.to(device)
        model.load_networks(opt.epoch)  #setup
        model.print_networks(verbose=False)

        return model, opt




    def load_cut_net(self):
        cut_model = create_model(self.opt)
        cut_model.setup(self.opt)
        cut_model.device='cuda:0'
        print(cut_model.device)

        # cut_model = cut_model.to(device)
        return cut_model

    def infer_cut_net(self, data):
        self.cut_model.set_input(data)  # unpack data from data loader
        self.cut_model.test()  # run inference
        visuals = self.cut_model.get_current_visuals()  # get image results
        # img_path = cut_model.get_image_paths()

        output_cut = visuals['fake_B']

        return output_cut


if __name__ == '__main__':
    cactuss = CACTUSS()

    print("Fetching test dataset...")
    #TEST_IMGS_A_GDRIVE_ID = '1yGODL5zZyUYLzKTFjA5yF_2s0pDK0YlR'
    #download_folder_from_gdrive(TEST_IMGS_A_GDRIVE_ID, DATASET_FOLDER + '/testA')
    #TEST_IMGS_B_GDRIVE_ID = '1jsmuUUGeW_IFFHyKarwF63rsKy4erNeR'
    #download_folder_from_gdrive(TEST_IMGS_B_GDRIVE_ID, DATASET_FOLDER + '/testB')

    dataset = create_dataset(cactuss.opt)
    #cut_model = cactuss.load_cut_net()

    if not os.path.exists(cactuss.opt.results_dir):
        os.mkdir(cactuss.opt.results_dir)

    inference_result_v = None

    for i, data in enumerate(dataset):
        # if i == 0:
        #     cut_model.data_dependent_initialize(data)
        #     cut_model.setup(cactuss.opt)  # model setup
        #     if cactuss.opt.eval:
        #         cut_model.eval()
        # if i >= cactuss.opt.num_test:  # only apply to opt.num_test images.
        #     break
        print(data['B'].shape)
       # input()
        output_cut = cactuss.infer_cut_net(data)
        output_cut_numpy = tensor2numpy(output_cut)
        orig_input = tensor2numpy(data['A'])

        output_cut = output_cut[:, 0, :, :].unsqueeze(0)

        # segm_result = cactuss.infer_seg_sim(output_cut[:, 0, :, :].unsqueeze(0))

        # inference_plot = np.hstack((orig_input[:, :, 0], output_cut_numpy[:, :, 0], segm_result * 255))
        inference_plot = output_cut
        if inference_result_v is None:
            inference_result_v = inference_plot
        else:
            inference_result_v = torch.vstack((inference_result_v, inference_plot))
        
        # show the results
        cv2.imshow('Inference result', (inference_plot.cpu().numpy()*255)[0,0,:,:])
        cv2.waitKey(0)

    cv2.imwrite(cactuss.opt.results_dir + 'inference_result_v1.png', inference_result_v.cpu().numpy())
    print('Inference complete, output can be found in' + cactuss.opt.results_dir  + ' folder.')

# cactuss = CACTUSS()
# US_dir="/home/aorta-scan//Amr/yavin/Manifold Learning/One_point_embedding/results/ceph_images"
# imgs_dir=sorted(os.listdir(US_dir))
# #cactuss.load_seg_net()
# #cactuss.load_cut_net()
# for img in iter(imgs_dir):
#     img=Image.open(os.path.join(US_dir,img))

#     img_plt=to_tensor(img)
#     img_plt=img_plt.repeat(1,3,1,1)
#     plt.subplot(1,3,1)
#     plt.imshow(img_plt[0,0,:,:])
#     #img=img_plt.to(device)



#     rgbimg = Image.new("RGB", img.size)
#     rgbimg.paste(img)
#     transform = get_transform(cactuss.cut_opt)

#     A = transform(rgbimg).to(device)
#     B = transform(cactuss.real_B).to(device)
#     A= torch.unsqueeze(A, 0)
#     print('A: ', A.shape)

#     data = {'A': A, 'B': B, 'A_paths': cactuss.a_paths, 'B_paths': cactuss.b_paths}


#    # img_dict={'A': img}
#    # img_dict['B']= img
#    # img_dict['A_paths']=DATAROOT_CUT + '/testA/p1_sweep2_9.png'
#    # img_dict['B_paths']=DATAROOT_CUT + '/testB/CT008_G4_label0008_all_filled_cropped_no_air_41_3.png'
    
#     output_cut = cactuss.infer_cut_net(data)
#     output_cut_numpy = tensor2numpy(output_cut)
#     print(output_cut_numpy.shape)
#     plt.subplot(1,3,2)
#     plt.imshow(torch.mean(output_cut,dim=1)[0,:,:].cpu())
#     print(torch.mean(output_cut,dim=1)[None,:,:,:].size())
#     segm_result = cactuss.infer_seg_sim(torch.mean(output_cut,dim=1)[:,None,:,:].to(device))
#     plt.subplot(1,3,3)
#     plt.imshow(segm_result)
#     plt.show()