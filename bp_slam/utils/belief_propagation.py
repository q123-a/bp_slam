"""
信念传播相关函数
Belief propagation related functions
"""

import numpy as np


def get_input_bp(existence, message_target_is_present):
    """
    结合锚点存在概率，计算输入信念传播消息

    参数:
        existence: 锚点存在的概率（标量，范围[0,1]）
        message_target_is_present: 当锚点存在时测量的消息向量，shape (len_message,)

    返回:
        message_out: 结合存在概率后的消息向量，用于信念传播
    """
    len_message = len(message_target_is_present)

    # 当锚点不存在时的消息，初始化为0向量，只有第一项（未检测）为1
    message_target_is_absent = np.zeros(len_message)
    message_target_is_absent[0] = 1

    # 根据存在概率加权合成消息
    # message_out = existence * message_target_is_present + (1 - existence) * message_target_is_absent
    # 体现了贝叶斯混合：锚点存在与否的加权消息
    message_out = existence * message_target_is_present + (1 - existence) * message_target_is_absent

    return message_out
