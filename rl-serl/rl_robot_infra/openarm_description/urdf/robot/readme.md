origin文件夹中是原始的xacro文件，外部我更改的只是制定了 openarm_description 的绝对路径
如此一来不需要安装ros2/ros的任何库

比较繁琐的是每个xacro中涉及到的地方都需要人工设置一下, 一旦遇到报错请耐心检查一下

涉及文件有 openarm_robot.xacro, v10.urdf.xacro, 以及其余include的子文件，凡是有 find robot_description 的地方都改

1. 转化为 urdf 执行下面的指令
xacro ./v10.urdf.xacro arm_type:=v10 bimanual:=true > openarm_bimanual.urdf

2. urdf 文件的路径重新调整，核心还是所有 mesh 均是 openarm_description package 路径下的
利用 vscode 中的 change all occurance 的功能，将mesh路径调整为绝对路径

