provider "aws" {
  region = var.region
}

variable "region" {
  description = "AWS region"
  default     = "eu-west-1"
}

resource "aws_s3_bucket" "docs" {
  bucket = "acme-search-docs"
}

resource "aws_lambda_function" "indexer" {
  function_name = "indexer"
  handler       = "searchapp.cli.main"
  environment {
    variables = {
      SEARCH_DB_PATH = "/tmp/index.db"
      BUCKET         = aws_s3_bucket.docs.bucket
    }
  }
}

module "net" {
  source = "./modules/net"
  cidr   = "10.0.0.0/16"
}

output "bucket" {
  value = aws_s3_bucket.docs.arn
}
