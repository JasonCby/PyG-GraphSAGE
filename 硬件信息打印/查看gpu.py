import nvsmi
import time
import psutil

# print(nvsmi.get_gpus())
# print(nvsmi.get_available_gpus())
# print(nvsmi.get_gpu_processes())
gpu_data = list(nvsmi.get_gpus())[0]
start_gpu_util = gpu_data.gpu_util
start_gpu_mem_use = gpu_data.mem_used
import psutil
import os

print('当前进程的内存使用：', psutil.Process(os.getpid()).memory_info().rss)
print('当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

# 1）每隔10秒记录一次磁盘读写累计量
# read_bytes = psutil.disk_io_counters().read_bytes
# wrtite_bytes = psutil.disk_io_counters().write_bytes
# total_bytes = read_bytes + wrtite_bytes
# # 2）减去上一次记录磁盘读写累计量，除以间隔时间（10秒）
# disk_bps = (total_bytes - total_bytes_last)/10

p = psutil.Process()
io_counters = p.io_counters()
disk_usage_process = io_counters[2] + io_counters[3]  # read_bytes + write_bytes
disk_io_counter = psutil.disk_io_counters()
disk_total = disk_io_counter[2] + disk_io_counter[3]  # read_bytes + write_bytes
print("Disk", disk_usage_process * 100 / disk_total)  # busy_time
psutil.cpu_stats()
while 1:
    time.sleep(1)
    # print(psutil.disk_io_counters().read_bytes/1024 / 1024 / 1024)
    # p = psutil.Process()
    io_counters = p.io_counters()
    disk_usage_process = io_counters[2] + io_counters[3]  # read_bytes + write_bytes
    disk_io_counter = psutil.disk_io_counters()
    disk_total = disk_io_counter[2] + disk_io_counter[3]  # read_bytes + write_bytes
    print("Disk", disk_usage_process * 100 / disk_total)
