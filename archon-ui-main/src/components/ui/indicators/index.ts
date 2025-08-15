/**
 * CompatibilityIndicator Components
 * 
 * Reusable visual indicators for model capabilities, performance, and status.
 * These components provide consistent visual feedback across the settings UI.
 */

export { ToolCallIndicator } from './ToolCallIndicator';
export { EmbeddingDimensionChip } from './EmbeddingDimensionChip';
export { PerformanceIndicator } from './PerformanceIndicator';
export { StatusBadge } from './StatusBadge';
export { CapabilityBadge } from './CapabilityBadge';
export { PricingIndicator } from './PricingIndicator';

// Re-export types for convenience
export type {
  ToolCallIndicatorProps,
  EmbeddingDimensionChipProps,
  PerformanceIndicatorProps,
  StatusBadgeProps,
  CapabilityBadgeProps,
  PricingIndicatorProps
} from './types';