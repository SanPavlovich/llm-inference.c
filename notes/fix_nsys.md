https://forums.developer.nvidia.com/t/nsys-doesnt-show-cuda-kernel-and-memory-data/315536/8

для wsl2 профилировщик nsys лагает и не показывает название кернелов и нет вкладки CUDA HW, поэтому нужно его починить в 3 команды

Find the Nsys config.ini file path from nsys -z. For example on my system:
1. nsys -z
/root/.config/NVIDIA Corporation/nsys-config.ini

2. Create the config.ini file if it does not already exist. Note the path might have a space in it so it needs to be wrapped by quotes
mkdir -p "/root/.config/NVIDIA Corporation"
touch "/home/liuyis/.config/NVIDIA Corporation/nsys-config.ini"

3. Add a line in the config file: CuptiUseRawGpuTimestamps=false
echo "CuptiUseRawGpuTimestamps=false" > "/root/.config/NVIDIA Corporation/nsys-config.ini"

После этого реально заработало!