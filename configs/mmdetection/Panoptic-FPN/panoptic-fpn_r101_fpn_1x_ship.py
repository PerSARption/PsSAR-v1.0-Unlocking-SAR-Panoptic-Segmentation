_base_ = './panoptic-fpn_r50_fpn_1x_ship_1.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))