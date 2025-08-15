/**
 * Type definitions for CompatibilityIndicator components
 */

export interface ToolCallIndicatorProps {
  supported: boolean;
  className?: string;
  showLabel?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export interface EmbeddingDimensionChipProps {
  dimensions: number;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
}

export interface PerformanceIndicatorProps {
  speed: 'fast' | 'medium' | 'slow';
  quality: 'high' | 'medium' | 'low';
  className?: string;
  layout?: 'horizontal' | 'vertical' | 'compact';
  showIcons?: boolean;
}

export interface StatusBadgeProps {
  status: 'available' | 'installing' | 'error' | 'unavailable';
  className?: string;
  showRetry?: boolean;
  onRetry?: () => void;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'badge' | 'indicator';
}

export interface CapabilityBadgeProps {
  capability: string;
  className?: string;
  type?: 'primary' | 'secondary' | 'success' | 'warning';
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
  description?: string;
}

export interface PricingIndicatorProps {
  pricing?: {
    input?: number;    // per token
    output?: number;   // per token
    unit?: string;     // usually 'per_token' or 'per_1k_tokens'
  };
  className?: string;
  layout?: 'horizontal' | 'vertical' | 'compact';
  showRelativeCost?: boolean;
  relativeCostLevel?: 1 | 2 | 3 | 4; // $ to $$$$
  showIcon?: boolean;
}