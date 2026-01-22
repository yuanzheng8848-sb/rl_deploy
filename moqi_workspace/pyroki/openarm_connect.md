# 连接顺序

# 控制电脑准备就绪
# 机械臂供电就绪
# 机械臂摆放处于零位
# 供电、Can线对其插入
# can-usb 接入电脑

# 接口映射
ls /dev | grep ttyACM
# 比如输出是 ttyACM0，表示当前连接的机械臂串口是 ttyACM0 这个值只与连接到电脑的先后有关（机械臂未供电也可以）
# 弄清楚usb对应的是哪个机械臂，即当前只插入一个usb，有一个串口

# 映射时，将对应的tyACM 与 can 建立联系，这里x是上一步得到，y是设定的（左0 右1）
sudo slcand -o -c -s8 /dev/ttyACMx cany

# 启用接口
sudo ip link set can0 up
sudo ip link set can1 up
