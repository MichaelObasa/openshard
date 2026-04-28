---
name: Terraform IaC Standards
description: Conventions and safety checks for Terraform infrastructure changes
category: infrastructure
keywords: [terraform, infra, iac, provider, module, state, plan, apply]
languages: []
framework: terraform
---

Always run terraform plan before apply; never apply without reviewing the diff.
Pin provider versions in required_providers — avoid >= without an upper bound.
Store state remotely (S3 + DynamoDB or Terraform Cloud); never commit .tfstate.
Use variable files (.tfvars) for environment-specific values, not hardcoded strings.
Tag all resources with environment, owner, and cost-centre for billing visibility.
