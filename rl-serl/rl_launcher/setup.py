from setuptools import setup, find_packages

setup(
    name="rl_launcher",
    version="0.1.0",
    description="RL network, agent, data, and launcher utilities for OpenArm rl-serl.",
    packages=find_packages(),
    install_requires=[
        "zmq",
        "typing",
        "typing_extensions",
        "opencv-python",
        "lz4",
        "agentlace@git+https://github.com/youliangtan/agentlace.git@cf2c337c5e3694cdbfc14831b239bd657bc4894d",
    ],
    zip_safe=False,
)
