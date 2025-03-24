from setuptools import find_packages, setup
from glob import glob

package_name = 'jaka_safe_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*')),
        ('share/' + package_name + '/logs', glob('logs/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lorenzo Busellato',
    maintainer_email='lorenzo.busellato@gmail.com',
    description='This package contains the implementations of safety-aware control architectures on the JAKA ZU 5 robot.',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vanilla_triangle_wave = jaka_safe_control.vanilla_triangle_wave:main',   
            'predictive_triangle_wave = jaka_safe_control.predictive_triangle_wave:main',          
            'leap_subscriber_node = jaka_safe_control.leap_subscriber_node:main'
        ],
    },
)
