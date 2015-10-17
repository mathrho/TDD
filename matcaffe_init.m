function net = matcaffe_init(use_gpu, model_def_file, model_file, gpu_id)
% matcaffe_init(model_def_file, model_file, use_gpu)
% Initilize matcaffe wrapper

if nargin < 1
  % By default use CPU
  use_gpu = 0;
end
if nargin < 2 || isempty(model_def_file)
  % By default use imagenet_deploy
  model_def_file = 'models/UCF_CNN_M_2048_deploy.prototxt';
end
if nargin < 3 || isempty(model_file)
  % By default use caffe reference model
  model_file = 'models/1_vgg_m_fine_tuning_rgb_iter_20000.caffemodel';
end

% set to use GPU or CPU
if use_gpu
  fprintf('Using GPU Mode\n');
  caffe.set_mode_gpu(); %caffe('set_mode_gpu');
else
  fprintf('Using CPU Mode\n');
  caffe.set_mode_cpu(); %caffe('set_mode_cpu');
end
fprintf('Done with set_mode\n');

%if caffe('is_initialized') == 0
if 1
  if exist(model_file, 'file') == 0
    % NOTE: you'll have to get the pre-trained ILSVRC network
    error('You need a network model file');
  end
  if ~exist(model_def_file,'file')
    % NOTE: you'll have to get network definition
    error('You need the network prototxt definition');
  end
  if nargin > 3
    caffe.set_device(gpu_id); %caffe('set_device',gpu_id);
  end
end

phase = 'test'; % run with phase test (so that dropout isn't applied)
net = caffe.Net(model_def_file, model_file, phase); %caffe('init', model_def_file, model_file)
fprintf('Done with init\n');

% put into test mode
%caffe.set_phase_test(); %caffe('set_phase_test');
fprintf('Done with set_phase_test\n');
