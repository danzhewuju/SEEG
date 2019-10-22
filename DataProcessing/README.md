# 数据预处理流程
1. pre_processing.py -> 主要是获取数据的分段， 读取文件的特定的时间区域

2. transferdata.py -> 主要是进行滤波，以及数据切片的处理，以及构造矩阵，将矩阵以文件的形式保存下来

3. data_process.py/data_duration_process.py 主要是用于训练集的数据划分，data_duration_process.py：预警时间的训练
data_process.py 三分类的数据划分

注意：需要设置patient_test 这个关键字， 之后所有的程序都依赖于这个关键字！


