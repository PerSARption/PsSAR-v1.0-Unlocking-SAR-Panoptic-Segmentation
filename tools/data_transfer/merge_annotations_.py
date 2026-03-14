#!/usr/bin/env python
"""
Merge ship annotations from instances_train2017.json into panoptic_train2017.json.
For each image, ship segments are appended to segments_info in the panoptic JSON.

Usage:
    python merge_annotations.py <panoptic_json> <instances_json> <output_json>

Example:
    python merge_annotations.py \
        annotations/panoptic_train2017.json \
        annotations/instances_train2017.json \
        annotations/panoptic_train2017_merged.json
"""

import json
import sys
import os


def main():
    if len(sys.argv) != 4:
        print("Usage: python merge_annotations.py "
              "<panoptic_json> <instances_json> <output_json>")
        sys.exit(1)

    panoptic_json  = sys.argv[1]
    instances_json = sys.argv[2]
    output_json    = sys.argv[3]

    with open(panoptic_json) as f:
        pan_data = json.load(f)
    with open(instances_json) as f:
        ins_data = json.load(f)

    # Build image_id -> list of ship segments from instances JSON
    # instances category_id=1 (ship), map back to panoptic category_id=0
    SHIP_PANOPTIC_CATEGORY_ID = 0
    ship_segments_by_image = {}
    for ann in ins_data['annotations']:
        image_id = ann['image_id']
        if image_id not in ship_segments_by_image:
            ship_segments_by_image[image_id] = []
        ship_segments_by_image[image_id].append(dict(
            id=ann['id'],                          # unique segment id
            category_id=SHIP_PANOPTIC_CATEGORY_ID, # ship=0 in panoptic
            iscrowd=ann['iscrowd'],
            bbox=ann['bbox'],
            area=ann['area'],
        ))

    # Make sure ship category is in panoptic categories
    cat_names = [c['name'] for c in pan_data['categories']]
    if 'ship' not in cat_names:
        pan_data['categories'].insert(0, dict(
            supercategory=None,
            isthing=1,
            id=SHIP_PANOPTIC_CATEGORY_ID,
            name='ship',
        ))
        print("Added ship category to panoptic JSON.")

    # Merge ship segments into each annotation's segments_info
    merged_count = 0
    for ann in pan_data['annotations']:
        image_id = ann['image_id']
        ship_segs = ship_segments_by_image.get(image_id, [])

        # Check for id conflicts with existing segments
        existing_ids = {seg['id'] for seg in ann['segments_info']}
        for ship_seg in ship_segs:
            # Resolve id conflict if any
            if ship_seg['id'] in existing_ids:
                new_id = max(existing_ids) + 1
                print(f"  [WARN] image_id={image_id}: "
                      f"id conflict {ship_seg['id']} -> {new_id}")
                ship_seg['id'] = new_id
            existing_ids.add(ship_seg['id'])
            ann['segments_info'].append(ship_seg)
            merged_count += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(pan_data, f)

    print(f"\nDone.")
    print(f"  Merged {merged_count} ship segments into panoptic JSON.")
    print(f"  Output saved to: {output_json}")

    # Verify
    images_with_ship = sum(
        1 for ann in pan_data['annotations']
        if any(s['category_id'] == SHIP_PANOPTIC_CATEGORY_ID
               for s in ann['segments_info'])
    )
    print(f"  Images with ship annotations: {images_with_ship} / "
          f"{len(pan_data['annotations'])}")


if __name__ == "__main__":
    main()
