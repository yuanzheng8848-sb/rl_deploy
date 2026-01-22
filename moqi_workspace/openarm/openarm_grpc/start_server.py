#!/usr/bin/env python3
"""
OpenArm gRPC 服务器启动脚本
"""

import sys
import os
import argparse

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from grpc_server import serve


def main():
    parser = argparse.ArgumentParser(description='启动 OpenArm gRPC 服务器')
    parser.add_argument('--port', type=int, default=50051, 
                       help='服务器端口 (默认: 50051)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='服务器地址 (默认: 0.0.0.0)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("OpenArm gRPC 服务器")
    print("=" * 50)
    print(f"服务器地址: {args.host}")
    print(f"服务器端口: {args.port}")
    print("=" * 50)
    
    try:
        serve(args.port)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
