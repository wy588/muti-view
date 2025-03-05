import numpy as np
import pandas as pd
import gc
import os

temp_dir = "/home/wangy/code/EDITS复现/multi-view/sorted_disk_data"
histogram_folder = "/home/wangy/code/EDITS复现/multi_view_mine/histogram_feature"


def sliding_window_disk_data(data, sw_width=30, max_leading_time=14, sliding_window_step=10):
    x = []
    tag = []
    level = []

    # 找到故障点（标签为1的位置）
    fault_indices = np.where(data[:, -1] == 1)[0]

    if len(fault_indices) > 0:
        for fault_index in fault_indices:
            # 对每个故障点，向前提取多个窗口
            end = fault_index + 1  # 包含故障点
            for j in range(max_leading_time):
                start = end - sw_width
                if start >= 0:
                    # 注意这里取的是特征列（不包括标签列）
                    slice = data[start:end, :-1]
                    if len(slice) == sw_width:  # 确保窗口长度正确
                        level.append(j + 1)  # 正样本难度级别
                        x.append(slice)
                        tag.append(1)
                end -= 1

    # 对于正常数据的采样
    start = 0
    while start + sw_width <= len(data):
        # 检查这个窗口内是否包含故障点
        if not any(data[start:start + sw_width, -1] == 1):  # 只有完全正常的窗口才标记为0
            slice = data[start:start + sw_width, :-1]
            x.append(slice)
            tag.append(0)
            level.append(-1)  # 负样本标记
        start += sliding_window_step

    x = np.array(x)
    tag = np.array(tag)
    level = np.array(level)

    return x, tag, level


# 获取所有硬盘的文件列表
disk_files = sorted(os.listdir(temp_dir))
histogram_files = sorted(os.listdir(histogram_folder))

# 从第58个批次开始处理
start_batch = 63
file_start_idx = start_batch * 1000  # 从第58000个文件开始
disk_files = disk_files[file_start_idx:]
histogram_files = histogram_files[file_start_idx:]

# 初始化存储变量时确保维度正确
data_k = None
histogram_k = None
tag = None
level = None
current_file_idx = file_start_idx

print(f"Starting from batch {start_batch}, file index {current_file_idx}")

for disk_file, hist_file in zip(disk_files, histogram_files):

    # 读取数据
    disk_data = pd.read_csv(f'{temp_dir}/{disk_file}', parse_dates=['dt']).set_index('dt').sort_index()
    hist_data = pd.read_csv(f'{histogram_folder}/{hist_file}')

    # 打印原始数据中的故障样本数
    fault_count = disk_data.iloc[:, -1].sum()


    if len(disk_data) != len(hist_data):
        print(f"长度不匹配: {disk_file} 和 {hist_file}, 跳过... {disk_data.shape[0]} vs {hist_data.shape[0]}")
        continue

    disk_array = np.delete(disk_data.to_numpy(), [0], axis=1)
    hist_array = hist_data.to_numpy()

    if len(disk_array) > 30:
        # 处理数据
        disk_temp, tag_temp, level_temp = sliding_window_disk_data(disk_array)
        hist_temp, _, _ = sliding_window_disk_data(hist_array)

        if disk_temp.shape[0] == hist_temp.shape[0]:
            # 打印更详细的调试信息
            if current_file_idx % 1000 == 0:
                print(f"Processing file {current_file_idx}")
                print(f"disk_temp shape: {disk_temp.shape}")
                print(f"disk_temp type: {type(disk_temp)}")
                print(f"data_k shape: {data_k.shape if data_k is not None else 'None'}")
                print(f"data_k type: {type(data_k) if data_k is not None else 'None'}")
            
            # 初始化或连接数据
            if data_k is None:
                data_k = disk_temp
                histogram_k = hist_temp
                tag = tag_temp
                level = level_temp
            else:
                try:
                    # 确保数组维度正确
                    disk_temp = np.array(disk_temp)
                    if len(disk_temp.shape) != 3:
                        print(f"Warning: disk_temp shape incorrect: {disk_temp.shape}")
                        continue
                        
                    # 连接数据
                    data_k = np.concatenate((data_k, disk_temp), axis=0)
                    histogram_k = np.concatenate((histogram_k, hist_temp), axis=0)
                    tag = np.concatenate((tag, tag_temp), axis=0)
                    level = np.concatenate((level, level_temp), axis=0)
                except Exception as e:
                    print(f"Error concatenating arrays: {str(e)}")
                    print(f"disk_temp shape: {disk_temp.shape}")
                    print(f"data_k shape: {data_k.shape}")
                    continue

        else:
            print(f"窗口化后数据维度不匹配: {disk_file} 和 {hist_file}, 跳过...")
            print(f"Dimensions: {disk_temp.shape} vs {hist_temp.shape}")
            continue

        current_file_idx += 1
        if current_file_idx % 1000 == 0:
            # 保存当前批次数据
            save_path = '/home/wangy/code/EDITS复现/multi_view_mine/sequence_feature_with_level'
            batch_num = current_file_idx // 1000

            np.save(f'{save_path}/raw_feature_sequence_{batch_num}', data_k)
            np.save(f'{save_path}/tag_{batch_num}', tag)
            np.save(f'{save_path}/level_{batch_num}', level)
            np.save(f'{save_path}/histogram_feature_sequence_{batch_num}', histogram_k)
            print(f"Saved batch {batch_num}")

            # 重置数组
            data_k = None
            histogram_k = None
            tag = None
            level = None

    gc.collect()

print("\nProcessing completed")
print("Final statistics:")
if tag is not None:
    print(f"Total samples processed: {len(tag)}")
    print(f"Total positive samples: {np.sum(tag == 1)}")
    print(f"Total negative samples: {np.sum(tag == 0)}")