-- Migration to change assignee column from enum to text
-- This allows any agent name to be assigned to tasks

BEGIN;

-- First, alter the column type from enum to text
ALTER TABLE tasks 
ALTER COLUMN assignee TYPE text 
USING assignee::text;

-- Set a default value
ALTER TABLE tasks 
ALTER COLUMN assignee SET DEFAULT 'User';

-- Add a check constraint to ensure assignee is not empty
ALTER TABLE tasks 
ADD CONSTRAINT assignee_not_empty CHECK (assignee IS NOT NULL AND assignee != '');

-- Drop the old enum type
DROP TYPE IF EXISTS task_assignee;

COMMIT;

-- Comment for documentation
COMMENT ON COLUMN tasks.assignee IS 'The agent or user assigned to this task. Can be any valid agent name or "User".';