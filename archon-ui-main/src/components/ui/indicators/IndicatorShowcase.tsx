import React from 'react';
import {
  ToolCallIndicator,
  EmbeddingDimensionChip,
  PerformanceIndicator,
  StatusBadge,
  CapabilityBadge,
  PricingIndicator
} from './index';

/**
 * Demo component showcasing all CompatibilityIndicator components
 * This component demonstrates various configurations and use cases
 */
export const IndicatorShowcase: React.FC = () => {
  return (
    <div className="p-6 space-y-8 bg-white dark:bg-gray-900">
      <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
        CompatibilityIndicator Components Showcase
      </div>
      
      {/* Tool Call Indicators */}
      <section className="space-y-4">
        <h3 className="text-lg font-semibold">Tool Call Indicators</h3>
        <div className="flex flex-wrap gap-4 p-4 border rounded-lg">
          <ToolCallIndicator supported={true} />
          <ToolCallIndicator supported={false} />
          <ToolCallIndicator supported={true} showLabel={false} size="sm" />
          <ToolCallIndicator supported={false} size="lg" />
        </div>
      </section>

      {/* Embedding Dimension Chips */}
      <section className="space-y-4">
        <h3 className="text-lg font-semibold">Embedding Dimension Chips</h3>
        <div className="flex flex-wrap gap-4 p-4 border rounded-lg">
          <EmbeddingDimensionChip dimensions={384} />
          <EmbeddingDimensionChip dimensions={768} />
          <EmbeddingDimensionChip dimensions={1536} />
          <EmbeddingDimensionChip dimensions={3072} />
          <EmbeddingDimensionChip dimensions={768} size="sm" showIcon={false} />
        </div>
      </section>

      {/* Performance Indicators */}
      <section className="space-y-4">
        <h3 className="text-lg font-semibold">Performance Indicators</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 border rounded-lg">
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Horizontal Layout</h4>
            <PerformanceIndicator speed="fast" quality="high" />
            <PerformanceIndicator speed="medium" quality="medium" />
            <PerformanceIndicator speed="slow" quality="high" />
          </div>
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Vertical Layout</h4>
            <PerformanceIndicator speed="fast" quality="low" layout="vertical" />
          </div>
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Compact Layout</h4>
            <PerformanceIndicator speed="fast" quality="high" layout="compact" />
            <PerformanceIndicator speed="medium" quality="medium" layout="compact" showIcons={false} />
          </div>
        </div>
      </section>

      {/* Status Badges */}
      <section className="space-y-4">
        <h3 className="text-lg font-semibold">Status Badges</h3>
        <div className="space-y-4 p-4 border rounded-lg">
          <div className="flex flex-wrap gap-4">
            <StatusBadge status="available" />
            <StatusBadge status="installing" />
            <StatusBadge status="error" showRetry onRetry={() => console.log('Retry clicked')} />
            <StatusBadge status="unavailable" />
          </div>
          <div className="flex flex-wrap gap-4">
            <StatusBadge status="available" variant="indicator" />
            <StatusBadge status="error" variant="indicator" showRetry onRetry={() => console.log('Retry clicked')} />
          </div>
        </div>
      </section>

      {/* Capability Badges */}
      <section className="space-y-4">
        <h3 className="text-lg font-semibold">Capability Badges</h3>
        <div className="flex flex-wrap gap-4 p-4 border rounded-lg">
          <CapabilityBadge capability="Vision" type="primary" />
          <CapabilityBadge capability="Function Calls" type="success" />
          <CapabilityBadge capability="Code Generation" type="secondary" />
          <CapabilityBadge capability="Multimodal" type="warning" />
          <CapabilityBadge capability="Embeddings" showIcon={false} />
          <CapabilityBadge capability="Reasoning" size="sm" />
        </div>
      </section>

      {/* Pricing Indicators */}
      <section className="space-y-4">
        <h3 className="text-lg font-semibold">Pricing Indicators</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4 border rounded-lg">
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Detailed Pricing</h4>
            <PricingIndicator 
              pricing={{ input: 0.0001, output: 0.0002, unit: 'per_token' }}
            />
            <PricingIndicator 
              pricing={{ input: 0.5, output: 1.5, unit: 'per_1k_tokens' }}
              layout="vertical"
            />
          </div>
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Compact Pricing</h4>
            <PricingIndicator 
              pricing={{ input: 0.0001, output: 0.0003 }}
              layout="compact"
            />
          </div>
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Relative Cost</h4>
            <PricingIndicator showRelativeCost relativeCostLevel={1} />
            <PricingIndicator showRelativeCost relativeCostLevel={2} />
            <PricingIndicator showRelativeCost relativeCostLevel={3} />
            <PricingIndicator showRelativeCost relativeCostLevel={4} />
          </div>
        </div>
      </section>

      {/* Combined Usage Example */}
      <section className="space-y-4">
        <h3 className="text-lg font-semibold">Combined Usage Example</h3>
        <div className="p-4 border rounded-lg bg-gray-50 dark:bg-gray-800">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="font-medium">GPT-4 Vision Preview</h4>
              <StatusBadge status="available" size="sm" />
            </div>
            
            <div className="flex flex-wrap gap-2">
              <CapabilityBadge capability="Vision" type="primary" size="sm" />
              <CapabilityBadge capability="Function Calls" type="success" size="sm" />
              <ToolCallIndicator supported={true} showLabel={false} size="sm" />
            </div>
            
            <div className="flex items-center justify-between">
              <PerformanceIndicator speed="medium" quality="high" layout="compact" />
              <PricingIndicator showRelativeCost relativeCostLevel={4} showIcon={false} />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};