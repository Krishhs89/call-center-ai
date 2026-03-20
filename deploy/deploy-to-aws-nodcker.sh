#!/usr/bin/env bash
# =============================================================================
# deploy-to-aws-nodcker.sh  —  AWS deployment WITHOUT local Docker
#
# Uses: GitHub repo → EC2 (git clone + Python 3.11 + systemd service)
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
GITHUB_REPO="${GITHUB_REPO:-https://github.com/Krishhs89/call-center-ai.git}"
GITHUB_BRANCH="${GITHUB_BRANCH:-master}"
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
    --branch)   GITHUB_BRANCH="$2"; shift 2 ;;
    *) warn "Unknown flag: $1"; shift ;;
  esac
done

# ── Step 0: Prerequisites ─────────────────────────────────────────────────────
step "Checking prerequisites"

command -v aws >/dev/null 2>&1 || error "AWS CLI not found. Run: brew install awscli"

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
    || error "No EC2 Key Pair found. Run: aws ec2 create-key-pair --key-name call-center-ai-key --query KeyMaterial --output text > ~/.ssh/call-center-ai-key.pem"
  warn "Auto-selected key pair: $KEY_PAIR_NAME"
fi

info "GitHub repo: $GITHUB_REPO (branch: $GITHUB_BRANCH)"
success "Prerequisites OK"

# ── Step 1: Store API keys in SSM ────────────────────────────────────────────
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
    warn "Skipping $name (empty in .env)"
  fi
}

store_param "ANTHROPIC_API_KEY" "${ANTHROPIC_API_KEY:-}"
store_param "OPENAI_API_KEY"    "${OPENAI_API_KEY:-}"
store_param "GOOGLE_API_KEY"    "${GOOGLE_API_KEY:-}"

# ── Step 2: Deploy CloudFormation ────────────────────────────────────────────
step "Deploying CloudFormation stack: $STACK_NAME"

CFN_TEMPLATE="$SCRIPT_DIR/cloudformation-ec2-python.yaml"
[[ -f "$CFN_TEMPLATE" ]] || error "Template not found: $CFN_TEMPLATE"

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
    ParameterKey=GitHubRepo,ParameterValue="$GITHUB_REPO" \
    ParameterKey=GitHubBranch,ParameterValue="$GITHUB_BRANCH" \
    ParameterKey=SSMPrefix,ParameterValue="$SSM_PREFIX" \
    ParameterKey=DefaultLLM,ParameterValue="$DEFAULT_LLM" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "$AWS_REGION" \
  --tags \
    Key=Project,Value=CallCenterAI \
    Key=ManagedBy,Value=deploy-nodcker-sh \
  2>&1 | grep -v "No updates are to be performed" || true

info "Waiting for stack to complete (3-5 minutes)..."
aws cloudformation wait "$WAIT_EVENT" \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  || warn "Wait timed out — check CloudFormation console for status"

# ── Step 3: Results ───────────────────────────────────────────────────────────
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
echo -e "  ${BOLD}GitHub:${NC}      ${GITHUB_REPO}"
echo -e "  ${BOLD}Stack:${NC}       ${STACK_NAME} (${AWS_REGION})"
echo ""
echo -e "  ${YELLOW}Note:${NC} App starts ~90 seconds after the instance boots."
echo -e "  If the URL doesn't load immediately, wait a minute and refresh."
echo ""
echo -e "  ${BOLD}SSH + logs:${NC}"
echo -e "  ssh -i <your-key>.pem ec2-user@${PUBLIC_IP}"
echo -e "  sudo journalctl -u call-center-ai -f"
echo ""
