#!/usr/bin/env bash
# =============================================================================
# deploy-to-aws-nodcker.sh  —  AWS deployment WITHOUT local Docker
#
# Uses: code ZIP → S3 → EC2 (Python 3.11 + systemd service)
#
# Usage:
#   ./deploy/deploy-to-aws-nodcker.sh [--region us-east-1] [--key-pair my-key]
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
KEY_PAIR_NAME="${KEY_PAIR_NAME:-}"
STACK_NAME="${STACK_NAME:-call-center-ai}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.small}"
DEFAULT_LLM="${DEFAULT_LLM:-claude}"
SSM_PREFIX="/call-center-ai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
step()    { echo -e "\n${BOLD}══ $* ══${NC}"; }

# ── Parse flags ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)   AWS_REGION="$2";    shift 2 ;;
    --key-pair) KEY_PAIR_NAME="$2"; shift 2 ;;
    --stack)    STACK_NAME="$2";    shift 2 ;;
    --instance) INSTANCE_TYPE="$2"; shift 2 ;;
    --llm)      DEFAULT_LLM="$2";   shift 2 ;;
    *) warn "Unknown flag: $1"; shift ;;
  esac
done

# ── Step 0: Prerequisites ─────────────────────────────────────────────────────
step "Checking prerequisites"

command -v aws  >/dev/null 2>&1 || error "AWS CLI not found. Run: brew install awscli"
command -v zip  >/dev/null 2>&1 || error "zip not found. Run: brew install zip"
command -v python3 >/dev/null 2>&1 || warn "python3 not found locally (only needed for local testing)"

aws sts get-caller-identity --region "$AWS_REGION" >/dev/null 2>&1 \
  || error "AWS credentials not configured. Run: aws configure"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
success "AWS account: $AWS_ACCOUNT_ID | Region: $AWS_REGION"

# Load .env
ENV_FILE="$PROJECT_DIR/.env"
[[ -f "$ENV_FILE" ]] || error ".env not found at $ENV_FILE"
set -o allexport
# shellcheck disable=SC1090
source "$ENV_FILE"
set +o allexport

[[ -n "${ANTHROPIC_API_KEY:-}" || -n "${OPENAI_API_KEY:-}" || -n "${GOOGLE_API_KEY:-}" ]] \
  || error "No API keys in .env"

# Auto-detect key pair
if [[ -z "$KEY_PAIR_NAME" ]]; then
  KEY_PAIR_NAME=$(aws ec2 describe-key-pairs \
    --query "KeyPairs[0].KeyName" --output text --region "$AWS_REGION" 2>/dev/null || echo "")
  [[ -n "$KEY_PAIR_NAME" && "$KEY_PAIR_NAME" != "None" ]] \
    || error "No EC2 Key Pair found. Create one: aws ec2 create-key-pair --key-name call-center-ai-key"
  warn "Auto-selected key pair: $KEY_PAIR_NAME"
fi

success "Prerequisites OK"

# ── Step 1: Create S3 bucket ──────────────────────────────────────────────────
step "Setting up S3 bucket"

S3_BUCKET="call-center-ai-deploy-${AWS_ACCOUNT_ID}-${AWS_REGION}"
S3_KEY="call_center_ai.zip"

# Create bucket if missing
aws s3api head-bucket --bucket "$S3_BUCKET" --region "$AWS_REGION" 2>/dev/null \
|| {
  info "Creating S3 bucket: $S3_BUCKET"
  if [[ "$AWS_REGION" == "us-east-1" ]]; then
    aws s3api create-bucket \
      --bucket "$S3_BUCKET" \
      --region "$AWS_REGION" >/dev/null
  else
    aws s3api create-bucket \
      --bucket "$S3_BUCKET" \
      --region "$AWS_REGION" \
      --create-bucket-configuration LocationConstraint="$AWS_REGION" >/dev/null
  fi
  # Block public access
  aws s3api put-public-access-block \
    --bucket "$S3_BUCKET" \
    --public-access-block-configuration \
      BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
}

success "S3 bucket: s3://$S3_BUCKET"

# ── Step 2: Package application ───────────────────────────────────────────────
step "Packaging application code"

cd "$PROJECT_DIR"
TMP_ZIP="/tmp/call_center_ai_$(date +%s).zip"

info "Creating application bundle (excluding venv, archive, .env)..."
zip -r "$TMP_ZIP" . \
  --exclude "venv/*" \
  --exclude ".env" \
  --exclude "archive/*" \
  --exclude "*.wav" \
  --exclude "*.mp3" \
  --exclude "*.pptx" \
  --exclude "*.docx" \
  --exclude "__pycache__/*" \
  --exclude "*.pyc" \
  --exclude ".git/*" \
  --exclude ".DS_Store" \
  --exclude "deploy/*" \
  --exclude "data/cache/*" \
  -q

ZIP_SIZE_MB=$(du -m "$TMP_ZIP" | cut -f1)
success "Bundle created: ${ZIP_SIZE_MB}MB at $TMP_ZIP"

# ── Step 3: Upload to S3 ─────────────────────────────────────────────────────
step "Uploading bundle to S3"

aws s3 cp "$TMP_ZIP" "s3://$S3_BUCKET/$S3_KEY" --region "$AWS_REGION"
rm "$TMP_ZIP"
success "Uploaded: s3://$S3_BUCKET/$S3_KEY"

# ── Step 4: Store API keys in SSM ────────────────────────────────────────────
step "Storing API keys in SSM Parameter Store"

store_param() {
  local name="$1"; local value="$2"
  if [[ -n "$value" ]]; then
    aws ssm put-parameter \
      --name "$SSM_PREFIX/$name" \
      --value "$value" \
      --type SecureString \
      --overwrite \
      --region "$AWS_REGION" >/dev/null
    success "Stored: $SSM_PREFIX/$name"
  else
    warn "Skipping $name (empty)"
  fi
}

store_param "ANTHROPIC_API_KEY" "${ANTHROPIC_API_KEY:-}"
store_param "OPENAI_API_KEY"    "${OPENAI_API_KEY:-}"
store_param "GOOGLE_API_KEY"    "${GOOGLE_API_KEY:-}"

# ── Step 5: Deploy CloudFormation ────────────────────────────────────────────
step "Deploying CloudFormation stack: $STACK_NAME"

CFN_TEMPLATE="$SCRIPT_DIR/cloudformation-ec2-python.yaml"

STACK_STATUS=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query "Stacks[0].StackStatus" \
  --output text 2>/dev/null || echo "DOES_NOT_EXIST")

if [[ "$STACK_STATUS" == "DOES_NOT_EXIST" ]]; then
  OPERATION="create-stack"; WAIT_EVENT="stack-create-complete"
  info "Creating new stack..."
else
  OPERATION="update-stack"; WAIT_EVENT="stack-update-complete"
  info "Updating stack (status: $STACK_STATUS)..."
fi

aws cloudformation "$OPERATION" \
  --stack-name "$STACK_NAME" \
  --template-body "file://$CFN_TEMPLATE" \
  --parameters \
    ParameterKey=InstanceType,ParameterValue="$INSTANCE_TYPE" \
    ParameterKey=KeyPairName,ParameterValue="$KEY_PAIR_NAME" \
    ParameterKey=S3BucketName,ParameterValue="$S3_BUCKET" \
    ParameterKey=S3ObjectKey,ParameterValue="$S3_KEY" \
    ParameterKey=SSMPrefix,ParameterValue="$SSM_PREFIX" \
    ParameterKey=DefaultLLM,ParameterValue="$DEFAULT_LLM" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "$AWS_REGION" \
  --tags \
    Key=Project,Value=CallCenterAI \
    Key=ManagedBy,Value=deploy-nodcker-sh \
  2>&1 | grep -v "No updates are to be performed" || true

info "Waiting for stack (3-5 minutes)..."
aws cloudformation wait "$WAIT_EVENT" \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  || warn "Wait timed out — check CloudFormation console"

# ── Step 6: Results ───────────────────────────────────────────────────────────
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
echo -e "  ${BOLD}Stack:${NC}       ${STACK_NAME} (${AWS_REGION})"
echo ""
echo -e "  ${YELLOW}Note:${NC} App starts ~90 seconds after the instance boots."
echo -e "  If the URL doesn't load, wait a minute and refresh."
echo ""
echo -e "  ${BOLD}SSH + logs:${NC}"
echo -e "  ssh -i <your-key>.pem ec2-user@${PUBLIC_IP}"
echo -e "  sudo journalctl -u call-center-ai -f"
echo ""
