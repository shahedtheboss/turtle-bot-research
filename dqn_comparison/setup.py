from setuptools import find_packages, setup

package_name = 'dqn_comparison'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    # v2: recorded obstacle paths used by worlds_v2.py (loaded next to the module)
    package_data={package_name: ['data/v2_obstacle_paths.csv']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shahed',
    maintainer_email='shahedshahadat0123@gmail.com',
    description='DQN / DDQN / D3QN / Rainbow comparison for TurtleBot3 navigation (ROS 2 Humble, Gazebo)',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={'console_scripts': []},
)
