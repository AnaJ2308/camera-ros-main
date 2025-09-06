import sys
if sys.prefix == '/home/eloisezeng/miniconda3':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/eloisezeng/ros2_ws/src/install/my_new_pkg'
