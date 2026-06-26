from setuptools import setup, find_packages

setup(
    name="rl_launcher",
    version="0.1.0",
    description="Thin forwarding layer over serl_launcher for OpenArm rl-serl.",
    packages=find_packages(),
    install_requires=[
        # serl_launcher itself is reused via sys.path (examples.compat) or
        # installed separately from hil-serl/serl_launcher.
    ],
    zip_safe=False,
)
