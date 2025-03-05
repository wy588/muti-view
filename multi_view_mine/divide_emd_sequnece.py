import numpy as np
import pandas as pd
import gc
import os

temp_dir = "/home/wangy/code/EDITS复现/multi-view/sorted_disk_data"
emd_folder="/home/wangy/code/EDITS复现/multi_view_mine/emd_feature"

def sliding_window_disk_data(data, sw_width=30, max_leading_time=14, sliding_window_step=10):
    x = []
    tag=[]

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
                    if len(slice) == sw_width:  # 确保窗口长度正确 # 正样本难度级别
                        x.append(slice)
                        tag.append(1)
                end -= 1

    # 对于正常数据的采样
    start = 0
    while start + sw_width <= len(data):
        # 检查这个窗口内是否包含故障点
        if not any(data[start:start + sw_width, -1] == 1):  # 只有完全正常的窗口才标记为0
            slice = data[start:start + sw_width, :-1]
            x.append(slice)# 负样本标记
            tag.append(0)
        start += sliding_window_step

    x = np.array(x)
    tag=np.array(tag)

    # 打印统计信息

    return x, tag


# 获取所有硬盘的文件列表
disk_files = sorted(os.listdir(temp_dir))
emd_files = sorted(os.listdir(emd_folder))

# 从第58个批次开始处理
start_batch = 19
file_start_idx = start_batch * 1000
disk_files = disk_files[file_start_idx:]
emd_files = emd_files[file_start_idx:]
# 初始化存储变量
data_k = None
emd_k = None
tag = None
i = 0

print(f"Total files to process: {len(disk_files)}")

for disk_file, emd_file in zip(disk_files, emd_files):

    # 读取数据
    disk_data = pd.read_csv(f'{temp_dir}/{disk_file}')
    emd_data=pd.read_csv(f'{emd_folder}/{emd_file}')

    # 打印原始数据中的故障样本数
    fault_count = disk_data.iloc[:, -1].sum()


    if len(disk_data) != len(emd_data):
        print(f"长度不匹配: {disk_file} 和 {emd_file}, 跳过... {disk_data.shape[0]} vs {emd_data.shape[0]}")
        continue

    disk_array = np.delete(disk_data.to_numpy(), [0], axis=1)
    emd_array = emd_data.to_numpy()

    if len(emd_array) > 30:
        # 处理数据
        disk_temp, tag_temp= sliding_window_disk_data(disk_array)
        emd_temp, _ = sliding_window_disk_data(emd_array)


        if disk_temp.shape[0] == emd_temp.shape[0]:
            # 第一批数据
            if i == 0:
                data_k = disk_temp
                emd_k = emd_temp
                tag = tag_temp

            else:
                # 连接数据
                data_k = np.concatenate((data_k, disk_temp), axis=0)
                emd_k = np.concatenate((emd_k, emd_temp), axis=0)
                tag = np.concatenate((tag, tag_temp), axis=0)


        else:
            print(f"窗口化后数据维度不匹配: {disk_file} 和 {emd_file}, 跳过...")
            print(f"Dimensions: {disk_temp.shape} vs {emd_temp.shape}")
            continue

        i += 1
        if i % 1000 == 0:
            # 保存当前批次数据
            save_path = '/home/wangy/code/EDITS复现/multi_view_mine/sequence_feature_with_level'
            batch_num = i // 1000



            # np.save(f'{save_path}/raw_feature_sequence_{batch_num}', data_k)
            # np.save(f'{save_path}/tag_{batch_num}', tag)

            np.save(f'{save_path}/emd_feature_sequence_{batch_num}', emd_k)
            print(f"Saved batch {batch_num}")

            # 重置数组
            data_k = np.empty((0, 30, disk_temp.shape[2]))
            emd_k = np.empty((0, 30, emd_temp.shape[2]))
            tag = np.array([])


    gc.collect()

print("\nProcessing completed")
print("Final statistics:")
if tag is not None:
    print(f"Total samples processed: {len(tag)}")
    print(f"Total positive samples: {np.sum(tag == 1)}")
    print(f"Total negative samples: {np.sum(tag == 0)}")