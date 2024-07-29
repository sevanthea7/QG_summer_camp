import os
from obs import ObsClient

ak = "F915WYG9INM7JCMKWYA8"
sk = "bWMR0xxcVBxOA6URk86efzREAOXLzoZvu6lkU00M"
endpoint = "https://obs.cn-south-1.myhuaweicloud.com"
bucket_name = "qg23onnx"
file_path = ".\Alex_leaf.zip"
object_key = 'Atlas/Alex_leaf.zip'


def upload_file(bucket_name, object_key, file_path, endpoint, ak, sk):
    """
    上传指定的OBS对象到本地文件夹。

    :param bucket_name: OBS桶名称
    :param object_key: OBS中对象的键
    :param file_path: 本地文件夹路径，上传文件的本地路径
    :param endpoint: OBS服务终端节点
    :param ak: 访问密钥ID
    :param sk: 密钥访问密钥
    """
    # 创建ObsClient实例
    obsClient = ObsClient(access_key_id=ak, secret_access_key=sk, server=endpoint)
    "https://obs.cn-south-1.myhuaweicloud.com"
    "https://qg23onnx.obs.cn-south-1.myhuaweicloud.com/output/return_image.jpg"
    # 构造本地文件的完整路径
    endpoint_tail = endpoint.split('https://')[-1]
    try:
        # 执行下载操作
        obsClient.putFile(bucket_name, object_key, file_path=file_path)
        url = "https://" + bucket_name + "." + endpoint_tail + "/" + object_key
        print(f"successful upload:{url}")
        return url
    except Exception as e:
        print(f"error：{e}")


upload_file( bucket_name, object_key, file_path, endpoint, ak, sk )