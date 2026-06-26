from setuptools import setup, find_packages

setup(
    name="rl_robot_infra",
    version="0.1.0",
    description="OpenArm robot infra (gym env + wrappers + flask server) for rl-serl.",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "opencv-python",
        "scipy",
        "requests",
        "flask",
        "numpy",
    ],
    zip_safe=False,
)
