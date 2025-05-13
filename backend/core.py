import boto3
import streamlit as st
import os
import uuid
from typing import List, Any, Dict

# AWS region env variable
os.environ["AWS_REGION"] = "us-central"