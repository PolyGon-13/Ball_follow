import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'yolo_project_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(where='src',exclude=['test']),
    package_dir={'':'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name,'launch'),glob(os.path.join('launch','*.launch.py'))),
        (os.path.join('share',package_name,'config'),glob(os.path.join('config','*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='polygon',
    maintainer_email='hyundiego@hanyang.ac.kr',
    description='Object detection and velocity estimation using YOLO',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_node=yolo_project_package.yolo_node:main',
            'vel_cal_node=yolo_project_package.vel_cal_node:main',
        ],
    },
)
