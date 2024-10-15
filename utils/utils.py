from shapely.geometry import box


def check_intersection(bbox1, bbox2):
    box1 = box(bbox1[0], bbox1[1], bbox1[2], bbox1[3])
    box2 = box(bbox2[0], bbox2[1], bbox2[2], bbox2[3])

    intersection = box1.intersection(box2)
    return intersection.area > 0
