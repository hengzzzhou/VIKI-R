import json
import numpy as np

def extract_points_from_data(data_data):
    """
    从data_data中提取点坐标
    """
    all_points = []
    for item in data_data:
        if 'regions' in item:
            for region in item['regions']:
                if region['shape_attributes']['name'] == 'point':
                    cx = region['shape_attributes']['cx']
                    cy = region['shape_attributes']['cy']
                    all_points.append([cx, cy])
    
    return np.array(all_points)

def group_points(points, group_size=5):
    """
    将点按照指定大小分组
    """
    n_points = len(points)
    n_groups = n_points // group_size
    groups = []
    
    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group = points[start_idx:end_idx].tolist()
        groups.append(group)
    
    # 处理剩余的点
    if n_points % group_size != 0:
        remaining = points[n_groups * group_size:].tolist()
        groups.append(remaining)
    
    return groups

def main():
    # 示例使用
    json_file = "/fs-computility/mabasic/zhouheng/work/embodied/verl/data/viki/viki_3/viki_3_3.json"
    with open(json_file, 'r') as f:
        data_data = json.load(f)
    
    points = extract_points_from_data(data_data)
    grouped_points = group_points(points)
    print(grouped_points)

if __name__ == "__main__":
    main() 