from setuptools import find_packages, setup

package_name = 'my_launch'

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
    maintainer='eloisezeng',
    maintainer_email='eloise.zeng@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest', 'pytest-cov',],
    entry_points={
        'console_scripts': [
            'heightfield_subscriber = my_launch.heightfield_subscriber:main',
            'live_subscriber = my_launch.live_subscriber:main',
            'pointcloud_subscriber = my_launch.pointcloud_subscriber:main',
            'multi_cam_subscriber = my_launch.multi_cam_subscriber:main',
    ],
},

)
