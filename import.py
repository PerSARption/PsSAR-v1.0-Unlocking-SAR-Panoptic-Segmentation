import json
import os

def check_labelme_annotations(annotation_dir):
    """
    检查LabelMe标注文件中多边形标注的顶点数量是否大于2个
    
    参数:
        annotation_dir: 存放LabelMe标注JSON文件的目录路径
    """
    # 检查目录是否存在
    if not os.path.exists(annotation_dir):
        print(f"错误: 目录 '{annotation_dir}' 不存在!")
        return
    
    # 统计问题文件数量
    error_count = 0
    
    # 遍历目录下所有JSON文件
    for filename in os.listdir(annotation_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(annotation_dir, filename)
            
            try:
                # 读取JSON文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查每个标注形状
                for shape in data.get('shapes', []):
                    # 只检查多边形类型的标注
                    if shape['shape_type'] == 'polygon':
                        points_count = len(shape['points'])
                        # 多边形需要至少3个顶点
                        if points_count < 3:
                            error_count += 1
                            print(f"问题文件: {filename}")
                            print(f"  标签: {shape['label']}")
                            print(f"  顶点数量: {points_count} (需要至少3个)\n")
            
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    
    if error_count == 0:
        print("所有标注文件检查通过，未发现顶点数量不足的多边形标注!")
    else:
        print(f"检查完成，共发现 {error_count} 处顶点数量不足的多边形标注，请及时修正。")

if __name__ == "__main__":
    # 标注文件所在目录
    annotation_directory = os.path.join(
        "mmdetection", 
        "tools", 
        "data_transfer", 
        "annotationed"
    )
    
    # 调用检查函数
    check_labelme_annotations(annotation_directory)
    