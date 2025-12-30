import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from config import Config

class S3Storage:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id= Config.AWS_ACCESS_KEY,
            aws_secret_access_key= Config.AWS_SECRET_KEY
        )
        self.bucket = Config.AWS_BUCKET_NAME

    def upload_file(self, file_obj,file_name):
        try:
            self.s3.upload_fileobj(file_obj, self.bucket, file_name)
            return True
        except ClientError as e:
            print(f"Client error: {e}")
            return False


    def get_file(self,file_name):
        try:
            file_obj = self.s3.get_object(Bucket=self.bucket, key=file_name)
            return file_obj['Body']
        except ClientError as e:
            print(f"Client error: {e}")
            return None        