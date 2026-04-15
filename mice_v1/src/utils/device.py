import torch


def detect_device():
    """
    自动检测可用设备并返回设备信息
    
    Returns:
        tuple: (device_str, device_info)
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        device_str = f"cuda:{current_device}"
        device_info = f"GPU: {device_name} (Device {current_device}/{device_count-1})"
        
        # 检查GPU内存
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"🚀 检测到GPU设备: {device_info}")
        print(f"   GPU内存: {memory_total:.1f}GB 总计, {memory_allocated:.1f}GB 已分配, {memory_reserved:.1f}GB 已预留")
        
    else:
        device_str = "cpu"
        device_info = "CPU"
        print("💻 未检测到可用GPU，使用CPU设备")
        
        # 检查CPU信息
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            memory_total = psutil.virtual_memory().total / 1024**3
            print(f"   CPU核心数: {cpu_count}, 系统内存: {memory_total:.1f}GB")
        except ImportError:
            print("   (安装psutil可查看详细CPU信息)")
    
    print(f"✅ 当前执行设备: {device_str}")
    return device_str, device_info