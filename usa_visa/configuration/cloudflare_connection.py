import boto3
import os
from usa_visa.constants import (
    R2_ACCESS_KEY_ID_ENV_KEY,
    R2_SECRET_ACCESS_KEY_ENV_KEY,
    R2_ENDPOINT_URL
)

class R2Client:
    r2_client = None
    r2_resource = None

    def __init__(self):
        """
        This class gets Cloudflare R2 credentials from environment variables and
        creates a connection with the R2 bucket.
        """
        if R2Client.r2_resource is None or R2Client.r2_client is None:
            __access_key_id = os.getenv(R2_ACCESS_KEY_ID_ENV_KEY)
            __secret_access_key = os.getenv(R2_SECRET_ACCESS_KEY_ENV_KEY)
            __endpoint_url = os.getenv(R2_ENDPOINT_URL)

            if __access_key_id is None:
                raise Exception(f"Environment variable: {R2_ACCESS_KEY_ID_ENV_KEY} is not set.")
            if __secret_access_key is None:
                raise Exception(f"Environment variable: {R2_SECRET_ACCESS_KEY_ENV_KEY} is not set.")
            if __endpoint_url is None:
                raise Exception(f"Environment variable: {R2_ENDPOINT_URL} is not set.")

            R2Client.r2_resource = boto3.resource(
                's3',
                aws_access_key_id=__access_key_id,
                aws_secret_access_key=__secret_access_key,
                endpoint_url=__endpoint_url
            )
            R2Client.r2_client = boto3.client(
                's3',
                aws_access_key_id=__access_key_id,
                aws_secret_access_key=__secret_access_key,
                endpoint_url=__endpoint_url
            )
        self.r2_resource = R2Client.r2_resource
        self.r2_client = R2Client.r2_client
