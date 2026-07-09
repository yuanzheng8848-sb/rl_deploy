# OpenArm CAN Migration

`openarm_can` is vendored at:

```text
rl-serl/third_party/openarm_can
```

The Python controller reads CAN settings from:

```text
rl-serl/rl_robot_infra/openarm_configs/can.yaml
```

## WSL Deployment Order

1. Attach the USB-CAN devices to WSL and confirm Linux can see them.

```bash
lsusb
ls /dev/ttyACM*
ip link
```

2. Install native build tools inside WSL.

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build pkg-config python3-dev
```

3. Build and install the C++ library plus Python binding.

```bash
cd rl-serl
bash scripts/build_openarm_can.sh
```

4. Configure SocketCAN.

For native SocketCAN devices:

```bash
bash scripts/configure_openarm_socketcan.sh can0
bash scripts/configure_openarm_socketcan.sh can1
```

For slcan/ttyACM adapters, create `can0`/`can1` first, then bring them up:

```bash
sudo slcand -o -c -s8 /dev/ttyACM0 can0
sudo slcand -o -c -s8 /dev/ttyACM1 can1
sudo ip link set can0 up
sudo ip link set can1 up
```

5. Update `openarm_configs/can.yaml` if WSL enumerates left/right differently.

6. Run import and bus checks before launching the full control server.

```bash
python scripts/check_openarm_can_import.py
python scripts/check_openarm_can_bus.py --side left
python scripts/check_openarm_can_bus.py --side right
```

Do not run the full `OpenArmController` until the CAN bus, side mapping, power,
and physical workspace are verified. The controller enables motors during
initialization.
