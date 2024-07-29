from mindcv.data import create_dataset, create_transforms, create_loader
import os

from download import download

cifar10_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"
# download(cifar10_url, "../data", kind="tar.gz", replace=True)
cifar10_dir = '../data/cifar-10-batches-bin'


num_classes = 10
num_workers = 1
download = not os.path.exists(cifar10_dir)


dataset_train = create_dataset(name='cifar10', root=cifar10_dir, split='train', shuffle=True, num_parallel_workers=num_workers, download=download)


trans = create_transforms(dataset_name='cifar10', image_resize=224)


loader_train = create_loader(dataset=dataset_train,
                             batch_size=64,
                             is_training=True,
                             num_classes=num_classes,
                             transform=trans,
                             num_parallel_workers=num_workers)

num_batches = loader_train.get_dataset_size()

from mindcv.models import create_model

network = create_model(model_name='densenet121', num_classes=num_classes, pretrained=True)

from mindcv.loss import create_loss

loss = create_loss(name='CE')

from mindcv.scheduler import create_scheduler


lr_scheduler = create_scheduler(steps_per_epoch=num_batches,
                                scheduler='constant',
                                lr=0.01)

from mindcv.optim import create_optimizer


opt = create_optimizer(network.trainable_params(), opt='adam', lr=lr_scheduler)

from mindspore import Model

model = Model(network, loss_fn=loss, optimizer=opt, metrics={'accuracy'})

from mindspore import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint

print( 'here' )
ckpt_save_dir = './ckpt'
ckpt_config = CheckpointConfig(save_checkpoint_steps=num_batches)
ckpt_cb = ModelCheckpoint(prefix='densenet121-cifar10',
                          directory=ckpt_save_dir,
                          config=ckpt_config)
print( 'here' )
model.train(5, loader_train, callbacks=[LossMonitor(num_batches//5), TimeMonitor(num_batches//5), ckpt_cb], dataset_sink_mode=False)


dataset_val = create_dataset(name='cifar10', root=cifar10_dir, split='test', shuffle=True, num_parallel_workers=num_workers, download=download)


loader_val = create_loader(dataset=dataset_val,
                           batch_size=64,
                           is_training=False,
                           num_classes=num_classes,
                           transform=trans,
                           num_parallel_workers=num_workers)


acc = model.eval(loader_val, dataset_sink_mode=False)
print(acc)