#!/usr/bin/env python
"""
Split a combined panoptic JSON (ship + sea + land) into:
  - panoptic_train2017.json  : sea + land only (stuff classes)
  - instances_train2017.json : ship only (thing class)

Usage:
    python split_annotations.py <input_json> <output_dir>

Example:
    python split_annotations.py panoptic_train2017_combined.json ./annotations/
"""

import json
import os
import sys

THING_CLASSES = ('ship',)
STUFF_CLASSES  = ('land', 'sea')


def main():
    if len(sys.argv) != 3:
        print("Usage: python split_annotations.py <input_json> <output_dir>")
        sys.exit(1)

    input_json = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json) as f:
        data = json.load(f)

    # Build category id → name map
    cat_id_to_name = {c['id']: c['name'] for c in data['categories']}

    # -----------------------------------------------------------------------
    # panoptic_train2017.json — sea + land only
    # -----------------------------------------------------------------------
    pan_categories = [
        c for c in data['categories'] if c['name'] in STUFF_CLASSES
    ]
    pan_annotations = []
    for ann in data['annotations']:
        stuff_segs = [
            seg for seg in ann['segments_info']
            if cat_id_to_name.get(seg['category_id']) in STUFF_CLASSES
        ]
        pan_annotations.append(dict(
            segments_info=stuff_segs,
            file_name=ann['file_name'],
            image_id=ann['image_id'],
        ))

    pan_data = dict(
        info=data['info'],
        licenses=data['licenses'],
        images=data['images'],
        type="instances",
        annotations=pan_annotations,
        categories=pan_categories,
    )

    pan_out = os.path.join(output_dir, "panoptic_train2017.json")
    with open(pan_out, 'w') as f:
        json.dump(pan_data, f)
    print(f"Saved panoptic JSON (sea+land): {pan_out}")

    # -----------------------------------------------------------------------
    # instances_train2017.json — ship only
    # ship category_id is 1 in instances format (per COCO convention)
    # -----------------------------------------------------------------------
    ins_categories = [dict(supercategory=None, id=1, name='ship')]

    ins_annotations = []
    ann_id = 0
    for ann in data['annotations']:
        image_id = ann['image_id']
        ship_segs = [
            seg for seg in ann['segments_info']
            if cat_id_to_name.get(seg['category_id']) in THING_CLASSES
        ]
        for seg in ship_segs:
            ins_annotations.append(dict(
                id=ann_id,
                image_id=image_id,
                category_id=1,
                segmentation=[],   # no polygon available from panoptic JSON
                area=seg['area'],
                bbox=seg['bbox'],
                iscrowd=seg['iscrowd'],
            ))
            ann_id += 1

    ins_data = dict(
        info=data['info'],
        licenses=data['licenses'],
        images=data['images'],
        type="instances",
        annotations=ins_annotations,
        categories=ins_categories,
    )

    ins_out = os.path.join(output_dir, "instances_train2017.json")
    with open(ins_out, 'w') as f:
        json.dump(ins_data, f)
    print(f"Saved instances JSON (ship): {ins_out}")

    # Summary
    print(f"\nDone.")
    print(f"  panoptic annotations : {len(pan_annotations)} images")
    print(f"  instances annotations: {len(ins_annotations)} ship instances")


if __name__ == "__main__":
    main()
