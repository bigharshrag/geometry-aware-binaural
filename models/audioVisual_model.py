import torch

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        #initialize model
        self.net_visual, self.net_audio, self.net_fusion, self.net_gen, self.net_classifier = nets

    def forward(self, data):
        visual_input = data['frame'].to(self.opt.device)
        audio_diff = data['audio_diff_spec'].to(self.opt.device)
        audio_mix = data['audio_mix_spec'].to(self.opt.device)

        audio_gt = audio_diff[:,:,:-1,:]

        input_spectrogram = audio_mix
        visual_feature, visual_feature_flat = self.net_visual(visual_input) 
        mask_prediction, upfeatures = self.net_audio(input_spectrogram, visual_feature_flat) 

        # complex masking to obtain the predicted spectrogram
        binaural_spectrogram = self.get_spectrogram(input_spectrogram, mask_prediction)

        # Fusion Network
        pred_left_mask, pred_right_mask = self.net_fusion(visual_feature, upfeatures) 
        fusion_left_spectrogram = self.get_spectrogram(input_spectrogram, pred_left_mask)
        fusion_right_spectrogram = self.get_spectrogram(input_spectrogram, pred_right_mask)

        output =  {'mask_prediction': mask_prediction, 'binaural_spectrogram': binaural_spectrogram, 'audio_gt': audio_gt,\
            'fusion_left_spectrogram': fusion_left_spectrogram, 'fusion_right_spectrogram': fusion_right_spectrogram,\
            'audio_mix': audio_mix}

        if self.opt.mode == 'train':
            if self.opt.use_geom_consistency:
                consistency_frame = data['consistency_frame'].to(self.opt.device)
            if self.opt.use_spatial_coherence:
                cl_spec = data['cl_spec'].to(self.opt.device) 
                label = data['label'].to(self.opt.device) 
            if self.opt.use_rir_pred:
                rir_spec = data['rir_spec'].to(self.opt.device)

            # Geomteric Consistency 
            if self.opt.use_geom_consistency:
                const_feat, _ = self.net_visual(consistency_frame)
                output.update({'const_feat': const_feat, 'vis_feat': visual_feature})

            # RIR Prediction
            if self.opt.use_rir_pred:
                pred_rir, _ = self.net_gen(visual_feature)
                output.update({'pred_rir': pred_rir, 'rir_spec': rir_spec})

            # Spatial Coherence
            if self.opt.use_spatial_coherence:
                pred = self.net_classifier(cl_spec, visual_feature)
                output.update({'cl_pred': pred, 'label': label})
            
        return output

    def get_spectrogram(self, input_spectrogram, mask_prediction):
        spectrogram_diff_real = input_spectrogram[:,0,:-1,:] * mask_prediction[:,0,:,:] - input_spectrogram[:,1,:-1,:] * mask_prediction[:,1,:,:]
        spectrogram_diff_img = input_spectrogram[:,0,:-1,:] * mask_prediction[:,1,:,:] + input_spectrogram[:,1,:-1,:] * mask_prediction[:,0,:,:]
        binaural_spectrogram = torch.cat((spectrogram_diff_real.unsqueeze(1), spectrogram_diff_img.unsqueeze(1)), 1)

        return binaural_spectrogram