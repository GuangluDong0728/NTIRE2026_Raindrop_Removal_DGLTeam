import os

def simple_rename_x4_suffix(folder_path):
    """
    简单版本：直接去掉文件名中的x4后缀
    
    Args:
        folder_path (str): LR图片文件夹路径
    """
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在！")
        return
    
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    renamed_count = 0
    
    print(f"开始处理文件夹：{folder_path}")
    print("-" * 50)
    
    for filename in files:
        # 检查文件是否包含x4
        if '_ALLINONE_TestLQ_FollowMedianTSE_1e_5_norotflip_p256_80000_meanWin16' in filename:
            # 构建完整的文件路径
            old_path = os.path.join(folder_path, filename)
            
            # 生成新文件名（去掉x4）
            new_filename = filename.replace('_ALLINONE_TestLQ_FollowMedianTSE_1e_5_norotflip_p256_80000_meanWin16', '')
            new_path = os.path.join(folder_path, new_filename)
            
            # 检查新文件名是否已存在
            if os.path.exists(new_path):
                print(f"跳过 {filename} -> 目标文件已存在")
                continue
            
            try:
                # 重命名文件
                os.rename(old_path, new_path)
                print(f"✓ {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"✗ 重命名失败 {filename}: {e}")
    
    print("-" * 50)
    print(f"完成！成功重命名 {renamed_count} 个文件")

# 使用方法：修改下面的路径为你的实际路径
if __name__ == "__main__":
    lr_folder = ""  # 修改为你的实际路径
    simple_rename_x4_suffix(lr_folder)
