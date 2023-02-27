import os
import time
import torch
from torch.nn.modules.loss import L1Loss
from options.train_options import TrainOptions
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from models.networks import Generator
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import torch.utils.data
from tensorboardX import SummaryWriter

from sklearn import manifold

def create_optimizer(nets, opt):
    (net_visual, net_audio, net_fusion, net_gen, net_classifier) = nets
    param_groups = [{'params': net_visual.parameters(), 'lr': opt.lr_visual},
                    {'params': net_audio.parameters(), 'lr': opt.lr_audio},
                    {'params': net_fusion.parameters(), 'lr': opt.lr_fusion}
                    ]
    if opt.use_spatial_coherence:
        param_groups.append({'params': net_classifier.parameters(), 'lr': opt.lr_cl})
    if opt.use_rir_pred:
        param_groups.append({'params': net_gen.parameters(), 'lr': opt.lr_gen})
    if opt.optimizer == 'sgd':
        return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def decrease_learning_rate(optimizer, decay_factor=0.94):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

#used to display validation loss
def display_val(model, loss_criterion, writer, index, dataset_val, opt):
    losses = []
    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            if i < opt.validation_batches:
                output = model(val_data)
                fusion_loss1 = loss_criterion(2*output['fusion_left_spectrogram']-output['audio_mix'][:,:,:-1,:], output['audio_gt'].detach())
                fusion_loss2 = loss_criterion(output['audio_mix'][:,:,:-1,:]-2*output['fusion_right_spectrogram'], output['audio_gt'].detach())
                fus_loss = (fusion_loss1 / 2 + fusion_loss2 / 2)
                loss = loss_criterion(output['binaural_spectrogram'], output['audio_gt'].detach()) + fus_loss

                losses.append(loss.item()) 
            else:
                break
    avg_loss = sum(losses)/len(losses)
    if opt.tensorboard:
        writer.add_scalar('data/val_loss', avg_loss, index)
    print('val loss: %.3f' % avg_loss)
    return avg_loss

#parse arguments
opt = TrainOptions().parse()
opt.device = torch.device("cuda")

def CreateDataset(opt):
    dataset = None
    if opt.model == 'audioVisual':
        from data.audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
    elif opt.model == 'fairPlay':
        from data.audioVisual_dataset import FairPlayDataset
        dataset = FairPlayDataset()
    elif opt.model == 'YoutubeBinaural':
        from data.audioVisual_dataset import YoutubeBinauralDataset
        dataset = YoutubeBinauralDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.model)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.nThreads))
    return dataset, dataloader

# construct data loader
ds, dataset = CreateDataset(opt)
dataset_size = len(ds)
print('#training clips = %d' % dataset_size)

#create validation set data loader if validation_on option is set
if opt.validation_on:
    #temperally set to val to load val data
    opt.mode = 'val'
    ds, _ = CreateDataset(opt)
    dataset_val = torch.utils.data.DataLoader(
        ds,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.nThreads))
    dataset_size_val = len(ds)
    print('#validation clips = %d' % dataset_size_val)
    opt.mode = 'train' #set it back

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(comment="_"+opt.name)
else:
    writer = None

# network builders
builder = ModelBuilder()
net_visual = builder.build_visual(weights=opt.weights_visual, map_location=opt.gpu_ids)
net_audio = builder.build_audio(
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
        weights=opt.weights_audio,
        map_location=opt.gpu_ids)
net_fusion = builder.build_fusion(weights=opt.weights_fusion, map_location=opt.gpu_ids)
nets = [net_visual, net_audio, net_fusion, None, None]

if opt.use_rir_pred:
    net_gen = builder.build_generator(weights=opt.weights_gen)
    nets[3] = net_gen
if opt.use_spatial_coherence:
    net_classifier = builder.build_classifier(weights=opt.weights_classifier)
    nets[4] = net_classifier

# construct our audio-visual model
model = AudioVisualModel(nets, opt)
model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model.to(opt.device)

# set up optimizer
optimizer = create_optimizer(nets, opt)

# set up loss function
loss_criterion = torch.nn.MSELoss()
rir_criterion = torch.nn.L1Loss()
spatial_coherence_criterion = torch.nn.BCEWithLogitsLoss()

# initialization
total_steps = 0
data_loading_time = []
model_forward_time = []
model_backward_time = []
batch_loss, batch_loss1, batch_fusion_loss, batch_rir_loss, batch_spat_const_loss, batch_geom_const_loss = [], [], [], [], [], []
best_err = float("inf")

for epoch in range(1, opt.niter+1):
    torch.cuda.synchronize()

    if(opt.measure_time):
        iter_start_time = time.time()
    
    for i, data in enumerate(dataset):
        if(opt.measure_time):
            torch.cuda.synchronize()
            iter_data_loaded_time = time.time()

        total_steps += opt.batchSize

        # forward pass
        optimizer.zero_grad()

        output = model(data)

        # compute loss
        loss1 = loss_criterion(output['binaural_spectrogram'], output['audio_gt'].detach())

        fusion_loss1 = loss_criterion(2*output['fusion_left_spectrogram']-output['audio_mix'][:,:,:-1,:], output['audio_gt'].detach())
        fusion_loss2 = loss_criterion(output['audio_mix'][:,:,:-1,:]-2*output['fusion_right_spectrogram'], output['audio_gt'].detach())
        fusion_loss = (fusion_loss1 / 2 + fusion_loss2 / 2)

        if opt.use_rir_pred:
            rir_loss = rir_criterion(output['rir_spec'], output['pred_rir'])
        if opt.use_spatial_coherence:
            spat_const_loss = spatial_coherence_criterion(output['cl_pred'], output['label'])
        if opt.use_geom_consistency:
            geom_const_loss = loss_criterion(output['vis_feat'], output['const_feat'])

        loss = opt.lambda0*loss1 + opt.lambda2*fusion_loss 

        if opt.use_rir_pred:
            loss += opt.lambda3*rir_loss
        if opt.use_spatial_coherence:
            loss += opt.lambda4*spat_const_loss
        if opt.use_geom_consistency:
            loss += opt.lambda5*geom_const_loss

        batch_loss.append(loss.item())
        batch_loss1.append(loss1.item())
        batch_fusion_loss.append(fusion_loss.item())
        if opt.use_rir_pred:
            batch_rir_loss.append(rir_loss.item())
        if opt.use_spatial_coherence:
            batch_spat_const_loss.append(spat_const_loss.item())
        if opt.use_geom_consistency:
            batch_geom_const_loss.append(geom_const_loss.item())

        if(opt.measure_time):
            torch.cuda.synchronize()
            iter_data_forwarded_time = time.time()

        loss.backward()
        optimizer.step()

        if(opt.measure_time):
            iter_model_backwarded_time = time.time()
            data_loading_time.append(iter_data_loaded_time - iter_start_time)
            model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
            model_backward_time.append(iter_model_backwarded_time - iter_data_forwarded_time)

        if(total_steps // opt.batchSize % opt.display_freq == 0):
            print('Display training progress at (epoch %d, total_steps %d)' % (epoch, total_steps))
            avg_loss = sum(batch_loss) / len(batch_loss)
            avg_loss1 = sum(batch_loss1) / len(batch_loss1)
            avg_fusion_loss = sum(batch_fusion_loss) / len(batch_fusion_loss)
            if opt.use_rir_pred:
                avg_rir_loss = sum(batch_rir_loss) / len(batch_rir_loss)
            if opt.use_spatial_coherence:
                avg_spat_const_loss = sum(batch_spat_const_loss) / len(batch_spat_const_loss)
            if opt.use_geom_consistency:
                avg_geom_const_loss = sum(batch_geom_const_loss) / len(batch_geom_const_loss)
            print('Average loss: %.3f' % (avg_loss))
            batch_loss, batch_loss1, batch_fusion_loss, batch_rir_loss, batch_spat_const_loss, batch_geom_const_loss = [], [], [], [], [], []
            if opt.tensorboard:
                writer.add_scalar('data/loss', avg_loss, total_steps)
                writer.add_scalar('data/loss1', avg_loss1, total_steps)
                writer.add_scalar('data/fusion_loss', avg_fusion_loss, total_steps)
                if opt.use_rir_pred:
                    writer.add_scalar('data/rir_loss', avg_rir_loss, total_steps)
                if opt.use_spatial_coherence:
                    writer.add_scalar('data/spat_const_loss', avg_spat_const_loss, total_steps)
                if opt.use_geom_consistency:
                    writer.add_scalar('data/geom_const_loss', avg_geom_const_loss, total_steps)
            if(opt.measure_time):
                print('average data loading time: ' + str(sum(data_loading_time)/len(data_loading_time)))
                print('average forward time: ' + str(sum(model_forward_time)/len(model_forward_time)))
                print('average backward time: ' + str(sum(model_backward_time)/len(model_backward_time)))
                data_loading_time = []
                model_forward_time = []
                model_backward_time = []
            print('end of display \n')

        if(total_steps // opt.batchSize % opt.save_latest_freq == 0):
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'visual_latest.pth'))
            torch.save(net_audio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'audio_latest.pth'))
            torch.save(net_fusion.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'fusion_latest.pth'))
            if opt.use_spatial_coherence:
                torch.save(net_classifier.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'classifier_latest.pth'))
            if opt.use_rir_pred:
                torch.save(net_gen.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'gen_latest.pth'))

        if(total_steps // opt.batchSize % opt.validation_freq == 0 and opt.validation_on):
            model.eval()
            opt.mode = 'val'
            print('Display validation results at (epoch %d, total_steps %d)' % (epoch, total_steps))
            val_err = display_val(model, loss_criterion, writer, total_steps, dataset_val, opt)
            print('end of display \n')
            model.train()
            opt.mode = 'train'
            #save the model that achieves the smallest validation error
            if val_err < best_err:
                best_err = val_err
                print('saving the best model (epoch %d, total_steps %d) with validation error %.3f\n' % (epoch, total_steps, val_err))
                torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'visual_best.pth'))
                torch.save(net_audio.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'audio_best.pth'))
                torch.save(net_fusion.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'fusion_best.pth'))
                if opt.use_spatial_coherence:
                    torch.save(net_classifier.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'classifier_best.pth'))
                if opt.use_rir_pred:
                    torch.save(net_gen.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'gen_best.pth'))
                        
            if opt.use_rir_pred:
                fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 4))
                img = librosa.display.specshow(librosa.amplitude_to_db(output['pred_rir'][0].cpu().detach().numpy()[0], ref=np.max), y_axis='log', x_axis='time', ax=ax[0][0])
                img = librosa.display.specshow(librosa.amplitude_to_db(output['rir_spec'][0].cpu().detach().numpy()[0], ref=np.max), y_axis='log', x_axis='time', ax=ax[0][1])
                img = librosa.display.specshow(librosa.amplitude_to_db(output['pred_rir'][0].cpu().detach().numpy()[1], ref=np.max), y_axis='log', x_axis='time', ax=ax[1][0])
                img = librosa.display.specshow(librosa.amplitude_to_db(output['rir_spec'][0].cpu().detach().numpy()[1], ref=np.max), y_axis='log', x_axis='time', ax=ax[1][1])
                fig.colorbar(img, ax=ax[0], format="%+2.0f dB")
                fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
                writer.add_figure('imgs/train', fig, total_steps)

        if(opt.measure_time):
            iter_start_time = time.time()

    #decrease learning rate 6% every opt.learning_rate_decrease_itr epochs
    if(opt.learning_rate_decrease_itr > 0 and epoch % opt.learning_rate_decrease_itr == 0):
        decrease_learning_rate(optimizer, opt.decay_factor)
        print('decreased learning rate by ', opt.decay_factor)
