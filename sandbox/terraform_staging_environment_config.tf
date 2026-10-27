terraform {
  required_version = ">= 1.6.0"
}

resource "aws_s3_bucket" "staging_artifacts" {
  bucket = "sortiq-staging-artifacts"
}