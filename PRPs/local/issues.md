## 1

2025-08-08 12:45:02 | search | WARNING | Failed to load storage settings: cannot access local variable 'credential_service' where it is not associated with a value, using defaults

## 2

there's an issue with the progress tracking after the restart. The logs show:

Before Restart:

- Processing batches sequentially: 8, 9, 10... of 24

After Restart:

The batch counter seems to continue but the context of total progress is unclear. This is because:

1. The crawl task is running in the background as an async task
2. When the service restarts, the task context is lost
3. The frontend reconnects but can't get accurate progress anymore

The Problem:

The crawl is actually still running (we can see it processing batches), but:

- The progress tracking is disconnected
- The frontend can't show accurate percentage
- The "of 24" total might be reset or lost

What's Happening:

1. The crawl task continues in memory/background
2. Code extraction is still processing (batches 8, 9, 10...)
3. WebSocket broadcasts continue
4. But the overall progress context (percentage, total items) is lost

This Explains:

- Why the frontend shows "crawl appears stuck" - it's not getting proper progress updates
- Why you see processing continue in logs but UI doesn't reflect it
- The disconnect between backend processing and frontend display

The crawl will likely complete successfully in the background, but the UI won't show accurate progress. You'll need to refresh the Knowledge Base page
after a few minutes to see if the new items were added.

This is a limitation of the current architecture - async tasks don't persist across service restarts.

## 3

**Add metrics tracking for embedding failures**

- Consider adding counters/metrics for different failure types (quota, rate limit, API errors)
- Would help monitor system health and identify patterns
