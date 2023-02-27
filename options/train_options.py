from .base_options import BaseOptions

class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--display_freq', type=int, default=50, help='frequency of displaying average loss')
		self.parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
		self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
		self.parser.add_argument('--niter', type=int, default=1000, help='# of epochs to train')
		self.parser.add_argument('--learning_rate_decrease_itr', type=int, default=-1, help='how often is the learning rate decreased by six percent')
		self.parser.add_argument('--decay_factor', type=float, default=0.94, help='learning rate decay factor')
		self.parser.add_argument('--tensorboard', type=bool, default=True, help='use tensorboard to visualize loss change ')		
		self.parser.add_argument('--measure_time', type=bool, default=False, help='measure time of different steps during training')
		self.parser.add_argument('--validation_on', action='store_true', help='whether to test on validation set during training')
		self.parser.add_argument('--validation_freq', type=int, default=100, help='frequency of testing on validation set')
		self.parser.add_argument('--validation_batches', type=int, default=100, help='number of batches to test for validation')
		self.parser.add_argument('--enable_data_augmentation', type=bool, default=True, help='whether to augment input frame')
		self.parser.add_argument('--use_spatial_coherence', action='store_true', help='whether to use the spatial coherence network')
		self.parser.add_argument('--use_geom_consistency', action='store_true', help='whether to use the geometric consistency network')
		self.parser.add_argument('--use_rir_pred', action='store_true', help='whether to use the rir prediction network')
		
		#model arguments
		self.parser.add_argument('--weights_visual', type=str, default='', help="weights for visual stream")
		self.parser.add_argument('--weights_audio', type=str, default='', help="weights for audio stream")
		self.parser.add_argument('--weights_encoder', type=str, default='', help="weights for encoder stream")
		self.parser.add_argument('--weights_fusion', type=str, default='', help="weights for fusion stream")
		self.parser.add_argument('--weights_classifier', type=str, default='', help="weights for classifier stream")
		self.parser.add_argument('--weights_gen', type=str, default='', help="weights for gen stream")
		self.parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
		self.parser.add_argument('--unet_input_nc', type=int, default=2, help="input spectrogram number of channels")
		self.parser.add_argument('--unet_output_nc', type=int, default=2, help="output spectrogram number of channels")

		#optimizer arguments
		self.parser.add_argument('--lr_visual', type=float, default=0.0001, help='learning rate for visual stream')
		self.parser.add_argument('--lr_audio', type=float, default=0.001, help='learning rate for unet')
		self.parser.add_argument('--lr_encoder', type=float, default=0.0001, help='learning rate for unet')
		self.parser.add_argument('--lr_fusion', type=float, default=0.001, help='learning rate for unet')
		self.parser.add_argument('--lr_cl', type=float, default=0.0001, help='learning rate for classifier')
		self.parser.add_argument('--lr_gen', type=float, default=0.0001, help='learning rate for rir gen')
		self.parser.add_argument('--optimizer', default='adam', type=str, help='adam or sgd for optimization')
		self.parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam')
		self.parser.add_argument('--weight_decay', default=0.0005, type=float, help='weights regularizer')
		self.parser.add_argument('--lambda0', default=1.0, type=float, help='Lambda for loss')
		self.parser.add_argument('--lambda1', default=0.0, type=float, help='Lambda for loss')
		self.parser.add_argument('--lambda2', default=1.0, type=float, help='Lambda for loss')
		self.parser.add_argument('--lambda3', default=1.0, type=float, help='Lambda for loss')
		self.parser.add_argument('--lambda4', default=1.0, type=float, help='Lambda for loss')
		self.parser.add_argument('--lambda5', default=1.0, type=float, help='Lambda for loss')

		self.mode = "train"
		self.isTrain = True
		self.enable_data_augmentation = True
