"""Check that the native openarm_can Python extension is importable."""

import openarm_can as oa


def main():
    print("openarm_can import OK")
    print("DM4310 enum:", oa.MotorType.DM4310)
    print("DM4340 enum:", oa.MotorType.DM4340)
    print("DM8009 enum:", oa.MotorType.DM8009)


if __name__ == "__main__":
    main()
