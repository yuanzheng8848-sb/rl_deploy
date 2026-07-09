import pyrealsense2 as rs
import cv2
import numpy as np
import time
# Create a pipeline to configure and start the camera

# 创建上下文对象
ctx = rs.context()
# 获取设备列表
devices = ctx.query_devices()

# 打印所有设备的序列号和名称，以便选择特定的设备
for dev in devices:
    print("Device name:", dev.get_info(rs.camera_info.name))
    print("Device serial number:", dev.get_info(rs.camera_info.serial_number))


class RealsenseCamera:
    def __init__(self, device_id=None, enable_depth=True, width=640, height=480, fps=30):
        """
        初始化 RealSense 相机

        Args:
            device_id: 相机序列号，None 则使用第一个设备
            enable_depth: 是否启用深度流
            width: 图像宽度
            height: 图像高度
            fps: 帧率
        """
        ctx = rs.context()
        # 获取设备列表
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise Exception("No Realsense device connected")
        # 打印所有设备的序列号和名称，以便选择特定的设备
        device = None
        if device_id is not None:
            for dev in devices:
                if dev.get_info(rs.camera_info.serial_number) == device_id:
                    device = dev
                    break
            if device is None:
                raise Exception(f"Device with serial number {device_id} not found")
        else:
            device = devices[0]

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(device.get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # 根据 enable_depth 参数决定是否启用深度流
        self.enable_depth = enable_depth
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.pipeline.start(config)
        self.rate = 0
        self.serial_number = device.get_info(rs.camera_info.serial_number)
        self.k=0
        # store last captured frames as instance attributes so other modules can access
        self.color_image = None
        self.depth_image = None

    def get_data(self, viz=False):
        s = time.perf_counter()
        data = [None, None]
        frames = self.pipeline.wait_for_frames()

        # RGB 图像处理
        rgb_frame = frames.get_color_frame()
        if rgb_frame:
            color_image = np.asanyarray(rgb_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            data[0] = color_image
            # expose on instance for DataRecorder or other modules
            self.color_image = color_image
            if viz:
                # cv2.imshow 期望 BGR 格式，所以需要转回 BGR 用于显示
                display_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                cv2.imshow(f'RGB Image-{self.serial_number}', display_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
        else:
            self.color_image = None

        # 深度图像处理（仅在启用时）
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                # convert depth frame to numpy array and expose
                depth_np = np.asanyarray(depth_frame.get_data())
                data[1] = depth_np
                self.depth_image = depth_np

                # 只在可视化时才进行颜色映射（这个操作很耗时）
                if viz:
                    depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_np, alpha=0.03), cv2.COLORMAP_JET)
                    cv2.imshow(f'depth Image-{self.serial_number}', depth_color)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        pass
            else:
                self.depth_image = None
        else:
            self.depth_image = None

        t = time.perf_counter()-s
        f = 1/t if t>0 else 0
        self.rate = (self.k*self.rate + f)/(self.k+1)
        self.k+=1

        return data



    def __del__(self):
        # self.pipeline.stop()
        pass

    def get_rate(self):
        return self.rate


class OpenCVCamera:
    def __init__(self, device_id=0, width=640, height=480, fps=30, exposure=None):
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Manually set exposure if provided
        if exposure is not None:
             # Disable auto exposure (1 = manual, 3 = auto)
             self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
             self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

        if not self.cap.isOpened():
            raise Exception(f"Could not open video device {device_id}")

    def get_data(self, viz=False):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # OpenCV returns BGR
        # If viz=True, we show it
        if viz:
            cv2.imshow("OpenCV Camera", frame)
            cv2.waitKey(1)

        return frame

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
