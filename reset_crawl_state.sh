#!/bin/bash
# reset_crawl_state.sh - Script to reset stalled crawl states in Redis
# Usage: ./reset_crawl_state.sh

set -e  # Exit on error

# Text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Redis keys
REDIS_KEY_RUNNING_FLAG="crawl:running"
REDIS_KEY_STATUS="crawl:status"

echo -e "${YELLOW}Checking Redis connection...${NC}"
if ! redis-cli ping > /dev/null 2>&1; then
  echo -e "${RED}Error: Cannot connect to Redis server.${NC}"
  echo "Please make sure Redis is running and try again."
  exit 1
fi
echo -e "${GREEN}Redis server is running.${NC}"

# Check if running flag exists
if redis-cli exists "${REDIS_KEY_RUNNING_FLAG}" | grep -q "1"; then
  echo -e "${YELLOW}Found running flag '${REDIS_KEY_RUNNING_FLAG}'.${NC}"
  # Delete the running flag
  redis-cli del "${REDIS_KEY_RUNNING_FLAG}"
  echo -e "${GREEN}Deleted running flag '${REDIS_KEY_RUNNING_FLAG}'.${NC}"
else
  echo -e "${GREEN}No running flag found.${NC}"
fi

# Check if status hash exists
if redis-cli exists "${REDIS_KEY_STATUS}" | grep -q "1"; then
  echo -e "${YELLOW}Found status hash '${REDIS_KEY_STATUS}'.${NC}"
  
  # Check current value of is_running
  is_running=$(redis-cli hget "${REDIS_KEY_STATUS}" "is_running")
  if [ "$is_running" = "1" ]; then
    echo -e "${YELLOW}Status shows crawl is still running. Resetting...${NC}"
    
    # Create ISO format date
    current_date=$(date -u +"%Y-%m-%dT%H:%M:%S.000000+00:00")
    
    # Update status hash
    redis-cli hset "${REDIS_KEY_STATUS}" "is_running" "0" "message" "Crawl reset manually" "end_time" "${current_date}"
    echo -e "${GREEN}Set status to not running with end time ${current_date}.${NC}"
  else
    echo -e "${GREEN}Status already shows crawl is not running.${NC}"
  fi
else
  echo -e "${YELLOW}No status hash found. Nothing to reset.${NC}"
fi

# Print final status
echo -e "\n${YELLOW}Current crawl status:${NC}"
redis-cli hgetall "${REDIS_KEY_STATUS}" | while read -r key; do
  read -r value
  echo -e "${YELLOW}$key:${NC} $value"
done

echo -e "\n${GREEN}Crawl state has been reset successfully.${NC}"
echo "You can now start a new crawl process." 