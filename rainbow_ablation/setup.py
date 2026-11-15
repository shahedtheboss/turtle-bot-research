from setuptools import find_packages, setup

package_name = 'rainbow_ablation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shahed',
    maintainer_email='shahedshahadat0123@gmail.com',
    description='Rainbow DQN guided-exploration ablation study (TurtleBot3, ROS 2 Humble, Gazebo)',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={'console_scripts': []},
)
