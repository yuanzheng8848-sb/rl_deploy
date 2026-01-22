CAN 的挂起

lsusb 看一下 DM 的设备是 ttyACM*
sudo slcand -o -c -s8 /dev/ttyACM0 can0
-s8 就是500M的速度了，看一下之前常进拉的和达妙的群

sudo ip link set can0 up

右手夹抓 闭合0.05 张开-1

-1 - 0.05

0 - 1
