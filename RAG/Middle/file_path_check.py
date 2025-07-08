import os

from Base.ModelException import ModelNotFoundError


def check_path(path: str, must_exist=True, is_file=True) -> str:
    """
    检查用户输入的路径：
    - 是否存在
    - 是不是文件或目录
    - 不存在或类型不符时抛出错误
    :param path: 文件/目录路径
    :param must_exist: 是否必须存在该文件/目录
    :param is_file: 是否为文件
    """
    if not isinstance(path, str):
        raise ValueError("路径必须是字符串")

    abs_path = os.path.abspath(path)

    if must_exist:
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"指定路径不存在：{abs_path}")
        if is_file and not os.path.isfile(abs_path):
            raise ValueError(f"路径不是有效文件：{abs_path}")
        if not is_file and not os.path.isdir(abs_path):
            raise ValueError(f"路径不是有效目录：{abs_path}")

    return abs_path


def check_model_path(path: str) -> str:
    """
    检查用户输入模型路径是否存在，不存在抛出异常

    :param path: 模型路径
    :return:
    """
    if not isinstance(path, str):
        raise ValueError("路径必须是字符串")

    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path) or not os.path.isdir(abs_path):
        raise ModelNotFoundError()

    return abs_path
