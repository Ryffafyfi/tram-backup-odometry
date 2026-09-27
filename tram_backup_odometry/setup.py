import os
from glob import glob

from setuptools import setup

package_name = 'tram_backup_odometry'


def files(pattern):
    return [f for f in glob(pattern) if os.path.isfile(f)]


setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', files('launch/*.launch.py')),
        ('share/' + package_name + '/config', files('config/*.yaml')),
        ('share/' + package_name + '/maps',
         files('maps/*.json') + files('maps/*.csv') + files('maps/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rafael',
    maintainer_email='ryffafyfi@gmail.com',
    description='Tram backup odometry (velocity + position from bogie speeds and driver controller)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node = tram_backup_odometry.node:main',
            'tram_backup_odometry = tram_backup_odometry.node:main',
            'perf_probe = tram_backup_odometry.perf_probe:main',
            'fault_player = tram_backup_odometry.fault_player:main',
        ],
    },
)
