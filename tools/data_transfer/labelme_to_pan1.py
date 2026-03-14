#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np

import labelme
from panopticapi import utils as ut

import cv2
import mmcv
import datetime
from mmengine.fileio import get
import uuid
import pycocotools.mask
import json
from PIL import Image

METAINFO = {
    'classes':
    ('ship', 'land', 'sea'),
    'thing_classes':
    ('ship', ),
    'stuff_classes':
    ('land', 'sea'),
    'palette':
    [(220, 20, 60), (119, 11, 32), (0, 0, 142)]
}


def draw_rectangle_safe(draw, xy, fill_ink=None):
    """Ensure that the rectangle coordinates are sorted correctly to avoid errors."""
    (x0, y0), (x1, y1) = xy
    # Ensure x1 >= x0 and y1 >= y0
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])
    draw.rectangle(((x0, y0), (x1, y1)), fill=fill_ink)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="Input annotated directory")
    parser.add_argument("output_dir", help="Output dataset directory")
    parser.add_argument(
        "--labels", help="Labels file or comma separated text", required=True
    )
    args = parser.parse_args()
    
    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "train2017"))
    os.makedirs(osp.join(args.output_dir, "panoptic_train2017"))
    os.makedirs(osp.join(args.output_dir, "panoptic_train2017_viz"))
    print("Creating dataset:", args.output_dir)
    out_ann_file = osp.join(args.output_dir, "panoptic_train2017.json")

    now = datetime.datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[
            dict(
                url=None,
                id=0,
                name=None,
            )
        ],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    if osp.exists(args.labels):
        with open(args.labels) as f:
            labels = [label.strip() for label in f if label]
    else:
        labels = [label.strip() for label in args.labels.split(",")]

    class_names = []
    class_name_to_id = {}
    for i, label in enumerate(labels):
        class_id = i - 1  # starts with -1
        class_name = label.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        if class_name in METAINFO['thing_classes']:
            data["categories"].append(
                dict(
                    supercategory=None,
                    isthing=1,
                    id=class_id,
                    name=class_name,
                )
            )
        elif class_name in METAINFO['stuff_classes']:
            data["categories"].append(
                dict(
                    supercategory=None,
                    isthing=0,
                    id=class_id,
                    name=class_name,
                )
            )
        else:
            print('The class_name is not predefined!')
            break

    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    for image_id, filename in enumerate(label_files):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)
        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "train2017", base + ".jpg")
        out_insp_file = osp.join(args.output_dir, "panoptic_train2017", base + ".png")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)

        # ------------------------------------------------------------------
        # FIX: build a unique-id map and save with lblsave so the PNG has
        # good visual contrast (same as original) while guaranteeing no
        # duplicate segment ids across categories.
        pan_seg = np.zeros(img.shape[:2], dtype=np.int32)
        segment_id = 1  # unique counter, incremented per segment
        # ------------------------------------------------------------------

        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=label_file.imagePath,
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        seg = dict(
            segments_info=[],
            file_name=base + ".png",
            image_id=image_id,
        )

        masks = {}
        for shape in label_file.shapes:

            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            print(f"Shape type: {shape_type}, Points: {points}")
            mask = labelme.utils.shape_to_mask(img.shape[:2], points, shape_type)

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)
            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                draw_rectangle_safe(
                    draw=cv2, xy=[(x1, y1), (x2, y2)], fill_ink=None
                )

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]
            if cls_id == -1:
                continue

            # FIX: assign unique segment_id directly instead of looking up
            # pixel values from pan_png (which caused id collisions).
            pan_seg[mask] = segment_id
            rgb_id = segment_id
            segment_id += 1

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            seg_instance = dict(
                id=int(rgb_id),
                category_id=cls_id,
                iscrowd=0,
                bbox=bbox,
                area=area
            )
            seg['segments_info'].append(seg_instance)

        # FIX: save with lblsave for good visual contrast, using our unique
        # pan_seg (each segment has a distinct id so no collisions occur)
        labelme.utils.lblsave(out_insp_file, pan_seg)

        # Read back the saved PNG to get the actual RGB-encoded ids
        # (lblsave applies a color mapping so we must reverse-lookup)
        pan_png = mmcv.imfrombytes(get(out_insp_file), flag='color', channel_order='rgb').squeeze()
        pan_png_ids = ut.rgb2id(pan_png)

        # Remap segment ids in segments_info to match what lblsave actually wrote
        for seg_info in seg['segments_info']:
            original_id = seg_info['id']
            # Find the actual encoded id by sampling pixels where pan_seg == original_id
            region = pan_png_ids[pan_seg == original_id]
            if len(region) > 0:
                unique_vals, counts = np.unique(region, return_counts=True)
                seg_info['id'] = int(unique_vals[np.argmax(counts)])

        # Save a high-contrast visualization PNG for human inspection
        # sea=red, land=blue, ship=cycling bright colors
        SHIP_COLORS = [
            (0, 255, 0), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (255, 128, 0), (128, 0, 255),
            (0, 128, 255), (255, 0, 128),
        ]
        cat_id_to_name = {c['id']: c['name'] for c in data['categories']}
        vis = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
        ship_color_idx = 0
        for seg_info in seg['segments_info']:
            cat_name = cat_id_to_name.get(seg_info['category_id'], '')
            if cat_name == 'sea':
                color = (255, 0, 0)
            elif cat_name == 'land':
                color = (0, 0, 255)
            elif cat_name == 'ship':
                color = SHIP_COLORS[ship_color_idx % len(SHIP_COLORS)]
                ship_color_idx += 1
            else:
                color = (128, 128, 128)
            vis[pan_seg == seg_info['id']] = color
        out_viz_file = osp.join(args.output_dir, "panoptic_train2017_viz", base + ".png")
        Image.fromarray(vis).save(out_viz_file)

        data['annotations'].append(seg)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
