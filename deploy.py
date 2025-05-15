import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

# Create a boto3 session with a region
boto_session = boto3.Session(region_name="us-east-2")
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# Specify the S3 path to your model.tar.gz
model_artifact = "s3://din-news-recommendation/models/din_model.tar.gz"
# IMPORTANT: Replace the role ARN with a valid SageMaker execution role ARN,
# not an Elasticache ARN.
role = "arn:aws:iam::915713862826:role/service-role/AmazonSageMaker-ExecutionRole-20250403T165574"

# Provide your Redis connection info from the previous step
redis_host = "news-uatdpu.serverless.use2.cache.amazonaws.com"
redis_port = "6379"

# Provide your VPC subnets and security group
vpc_config = {
    "Subnets": ["subnet-0bba66496801cc5b0", "subnet-03ffd5ae70f486658", "subnet-0a8778d16f1f0ea48"],
    "SecurityGroupIds": ["sg-0e5481ca1ce449de7"]
}

# Create the PyTorchModel
pytorch_model = PyTorchModel(
    model_data=model_artifact,
    entry_point="inference.py",
    source_dir="code",
    framework_version="1.12",
    py_version="py38",
    role=role,
    env={"REDIS_HOST": redis_host, "REDIS_PORT": redis_port},
    vpc_config=vpc_config,
    sagemaker_session=sagemaker_session  # Pass the session with region info
)

# Deploy the model to a real-time HTTPS endpoint
endpoint_name = "din-recommender-endpoint"
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name=endpoint_name
)
