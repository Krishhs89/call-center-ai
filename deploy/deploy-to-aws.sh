#!/usr/bin/env bash
# =============================================================================
# deploy-to-aws.sh  —  One-command AWS deployment for Call Center AI
#
# Usage:
#   ./deploy/deploy-to-aws.sh [--region us-east-1] [--key-pair my-keypair]
#
# What this script does:
#   1.  Validates prerequisites (AWS CLI, Docker, .env)
#   2.  Creates an ECR repository (if not already present)
#   3.  Builds the Docker image and pushes it to ECR
#   4.  Stores API keys securely in SSM Parameter Store
#   5.  Deploys (or updates) a CloudFormation stack with EC2 + Elastic IP
#   6.  Prints the live public URL
# =============================================================================

set -euo pipefail

# ── Defaults (override via flags or environment variables) ───────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
KEY_PAIR_NAME="${KEY_PAIR_NAME:-}"
STACK_NAME="${STACK_NAME:-call-center-ai}"
ECR_REPO_NAME="${ECR_REPO_NAME:-call-center-ai}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.small}"
DEFAULT_LLM="${DEFAULT_LLM:-claude}"
SSM_PREFIX="/call-center-ai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Colour helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
step()    { echo -e "\n${BOLD}══ $* ══${NC}"; }

# ── Parse CLI flags ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)     AWS_REGION="$2";      shift 2 ;;
    --key-pair)   KEY_PAIR_NAME="$2";   shift 2 ;;
    --stack)      STACK_NAME="$2";      shift 2 ;;
    --instance)   INSTANCE_TYPE="$2";   shift 2 ;;
    --tag)        IMAGE_TAG="$2";       shift 2 ;;
    --llm)        DEFAULT_LLM="$2";     shift 2 ;;
    *) warn "Unknown flag: $1"; shift ;;
  esac
done

# ── Step 0: Validate prerequisites ───────────────────────────────────────────
step "Validating prerequisites"

command -v aws    >/dev/null 2>&1 || error "AWS CLI not found. Install: brew install awscli"
command -v docker >/dev/null 2>&1 || error "Docker not found. Install Docker Desktop."

docker info >/dev/null 2>&1 || error "Docker daemon is not running. Start Docker Desktop."

# Check AWS credentials
aws sts get-caller-identity --region "$AWS_REGION" >/dev/null 2>&1 \
  || error "AWS credentials not configured. Run: aws configure"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
success "AWS account: $AWS_ACCOUNT_ID | Region: $AWS_REGION"

# Check .env file
ENV_FILE="$PROJECT_DIR/.env"
[[ -f "$ENV_FILE" ]] || error ".env file not found at $ENV_FILE"

# Source .env (non-exported variables)
set -o allexport
# shellcheck disable=SC1090
source "$ENV_FILE"
set +o allexport

# Require at least one LLM key
[[ -n "${ANTHROPIC_API_KEY:-}" || -n "${OPENAI_API_KEY:-}" || -n "${GOOGLE_API_KEY:-}" ]] \
  || error "No API keys found in .env. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY."

# Require a key pair name for SSH
if [[ -z "$KEY_PAIR_NAME" ]]; then
  # Try to find one automatically
  KEY_PAIR_NAME=$(aws ec2 describe-key-pairs \
    --query "KeyPairs[0].KeyName" --output text --region "$AWS_REGION" 2>/dev/null || echo "")
  [[ -n "$KEY_PAIR_NAME" && "$KEY_PAIR_NAME" != "None" ]] \
    || error "No EC2 Key Pair found. Create one in the AWS console and pass --key-pair <name>"
  warn "Using auto-detected key pair: $KEY_PAIR_NAME"
fi

success "Prerequisites OK"

# ── Step 1: Create ECR repository ────────────────────────────────────────────
step "Setting up ECR repository"

ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
FULL_IMAGE_URI="$ECR_URI/$ECR_REPO_NAME:$IMAGE_TAG"

# Create repo if it doesn't exist
aws ecr describe-repositories \
  --repository-names "$ECR_REPO_NAME" \
  --region "$AWS_REGION" >/dev/null 2>&1 \
|| {
  info "Creating ECR repository: $ECR_REPO_NAME"
  aws ecr create-repository \
    --repository-name "$ECR_REPO_NAME" \
    --image-scanning-configuration scanOnPush=true \
    --region "$AWS_REGION" >/dev/null
}

success "ECR repository: $FULL_IMAGE_URI"

# ── Step 2: Build and push Docker image ──────────────────────────────────────
step "Building Docker image"

cd "$PROJECT_DIR"

info "Building image (this may take 2-4 minutes on first run)..."
docker build \
  --platform linux/amd64 \
  -t "$ECR_REPO_NAME:$IMAGE_TAG" \
  -t "$FULL_IMAGE_URI" \
  .

success "Image built: $ECR_REPO_NAME:$IMAGE_TAG"

step "Pushing image to ECR"

info "Authenticating Docker with ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_URI"

info "Pushing image (may take a few minutes)..."
docker push "$FULL_IMAGE_URI"

success "Image pushed: $FULL_IMAGE_URI"

# ── Step 3: Store API keys in SSM Parameter Store ────────────────────────────
step "Storing API keys in SSM Parameter Store"

store_param() {
  local name="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    aws ssm put-parameter \
      --name "$SSM_PREFIX/$name" \
      --value "$value" \
      --type SecureString \
      --overwrite \
      --region "$AWS_REGION" >/dev/null
    success "Stored: $SSM_PREFIX/$name"
  else
    warn "Skipping $name (not set in .env)"
  fi
}

store_param "ANTHROPIC_API_KEY" "${ANTHROPIC_API_KEY:-}"
store_param "OPENAI_API_KEY"    "${OPENAI_API_KEY:-}"
store_param "GOOGLE_API_KEY"    "${GOOGLE_API_KEY:-}"

# ── Step 4: Deploy CloudFormation stack ──────────────────────────────────────
step "Deploying CloudFormation stack: $STACK_NAME"

CFN_TEMPLATE="$SCRIPT_DIR/cloudformation.yaml"
[[ -f "$CFN_TEMPLATE" ]] || error "CloudFormation template not found: $CFN_TEMPLATE"

# Check if stack already exists
STACK_STATUS=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query "Stacks[0].StackStatus" \
  --output text 2>/dev/null || echo "DOES_NOT_EXIST")

if [[ "$STACK_STATUS" == "DOES_NOT_EXIST" ]]; then
  info "Creating new stack..."
  OPERATION="create-stack"
  WAIT_EVENT="stack-create-complete"
else
  info "Updating existing stack (status: $STACK_STATUS)..."
  OPERATION="update-stack"
  WAIT_EVENT="stack-update-complete"
fi

aws cloudformation "$OPERATION" \
  --stack-name "$STACK_NAME" \
  --template-body "file://$CFN_TEMPLATE" \
  --parameters \
    ParameterKey=InstanceType,ParameterValue="$INSTANCE_TYPE" \
    ParameterKey=KeyPairName,ParameterValue="$KEY_PAIR_NAME" \
    ParameterKey=ECRImageURI,ParameterValue="$FULL_IMAGE_URI" \
    ParameterKey=SSMPrefix,ParameterValue="$SSM_PREFIX" \
    ParameterKey=DefaultLLM,ParameterValue="$DEFAULT_LLM" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "$AWS_REGION" \
  --tags \
    Key=Project,Value=CallCenterAI \
    Key=ManagedBy,Value=deploy-to-aws-sh \
  2>&1 | grep -v "No updates are to be performed" || true

info "Waiting for stack to complete (typically 3-5 minutes)..."
aws cloudformation wait "$WAIT_EVENT" \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  || warn "Stack wait timed out — check CloudFormation console for status"

# ── Step 5: Print results ─────────────────────────────────────────────────────
step "Deployment complete"

APP_URL=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='AppURL'].OutputValue" \
  --output text 2>/dev/null || echo "")

PUBLIC_IP=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='PublicIP'].OutputValue" \
  --output text 2>/dev/null || echo "")

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Call Center AI — Deployed Successfully!          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}App URL:${NC}     ${BLUE}${APP_URL}${NC}"
echo -e "  ${BOLD}Public IP:${NC}   ${PUBLIC_IP}"
echo -e "  ${BOLD}ECR Image:${NC}   ${FULL_IMAGE_URI}"
echo -e "  ${BOLD}Stack:${NC}       ${STACK_NAME} (${AWS_REGION})"
echo ""
echo -e "  ${YELLOW}Note:${NC} The app takes ~60 seconds to start after the instance boots."
echo -e "  If the URL doesn't load immediately, wait a minute and refresh."
echo ""
echo -e "  ${BOLD}SSH access:${NC}"
echo -e "  ssh -i <your-key>.pem ec2-user@${PUBLIC_IP}"
echo -e "  docker logs -f call-center-ai"
echo ""
