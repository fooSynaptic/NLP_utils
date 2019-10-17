import oss2
import json
from clip_news_v2.log import logger

class OSSOperate:
    def __init__(self, access_key_id, access_key_secret, endpoint, bucket_name):
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.file_queue = []
        self.oss = None
        self.bucket = None
        self.id = 0
        self.stop = False
        self.queueid = 0
        self.on_upload_complete = None

    def download_file(self, key, file):
        """
        将阿里云的文件下载到本地
        :param key:
        :param file:
        :return:
        """
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
        # bucket.get_object_to_file(key, file)
        try:
            # key = file.split("/")[-1]
            bucket.get_object_to_file(key, file)
        #     logger.info("download %s successed" % key)
        except Exception as e:
            return False
        #     logger.error("download failed")
        #     logger.error(e)

        return True

    def upload_file(self, key, file, headers=None):
        """
        上传本地文件至阿里云
        :param file:
        :return:
        """
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        # service = oss2.Service(auth,endpoint)
        bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
        try:
            # key = file.split("/")[-1]
            bucket.put_object_from_file(key, file, headers=headers)
            logger.info("upload %s successed" % key)
        except Exception as e:
            logger.error("upload failed")
            logger.error(e)
        return True

    def is_file_exist(self, file):
        """
        判断文件是否存在
        :param file:
        :return:
        """
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
        if bucket.object_exists(file):
            return True
        else:
            return False

    def delete_file(self, file):
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
        try:
            # if self.is_file_exist(file):
            bucket.delete_object(file)
            return True
            # else:
            #     logger.warning('delete failed: %s is not exist' % file)
            #     return False
        except Exception as e:
            logger.error(e)
            logger.error('delete %s failed' % file)
            return False

    def list_files(self, prefix="", marker="", delimiter=""):
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
        try:
            return oss2.ObjectIterator(bucket, prefix=prefix, marker=marker, delimiter=delimiter)
        except Exception as e:
            logger.error(e)
            return None

    def remote_stream(self, oss_key):
        """
        流式下载
        :param oss_key:
        :return:
        """
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
        try:
            return bucket.get_object(oss_key).read()
        except Exception as e:
            logger.error(e)
            return None

    def remote_upload_stream(self, oss_key, text):
        """
        流式上传
        :param oss_key:
        :param text:
        :return:
        """
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
        try:
            bucket.put_object(oss_key, json.dumps(text, ensure_ascii=False))
        except Exception as e:
            print(e)
            return False
