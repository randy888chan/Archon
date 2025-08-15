# CompatibilityIndicator Components

A comprehensive collection of visual indicators for model capabilities, performance metrics, and status information in the Archon UI.

## Components

### 1. ToolCallIndicator

Visual badge showing if a model supports function/tool calling.

```tsx
<ToolCallIndicator supported={true} />
<ToolCallIndicator supported={false} showLabel={false} size="sm" />
```

**Props:**
- `supported: boolean` - Whether the model supports function/tool calling
- `className?: string` - Additional CSS classes
- `showLabel?: boolean` - Show text label (default: true)
- `size?: 'sm' | 'md' | 'lg'` - Size variant (default: 'md')

### 2. EmbeddingDimensionChip

Chip showing embedding dimensions with color-coded performance characteristics.

```tsx
<EmbeddingDimensionChip dimensions={768} />
<EmbeddingDimensionChip dimensions={1536} size="lg" showIcon={false} />
```

**Props:**
- `dimensions: number` - Embedding dimension count
- `className?: string` - Additional CSS classes
- `size?: 'sm' | 'md' | 'lg'` - Size variant (default: 'md')
- `showIcon?: boolean` - Show layers icon (default: true)

**Color Coding:**
- Green (≤ 384): Fast, lightweight embeddings
- Blue (≤ 768): Balanced speed and quality
- Purple (≤ 1536): High quality embeddings
- Orange (> 1536): Maximum quality, slower processing

### 3. PerformanceIndicator

Shows speed vs quality indicators with multiple layout options.

```tsx
<PerformanceIndicator speed="fast" quality="high" />
<PerformanceIndicator speed="medium" quality="medium" layout="vertical" />
<PerformanceIndicator speed="slow" quality="high" layout="compact" />
```

**Props:**
- `speed: 'fast' | 'medium' | 'slow'` - Processing speed
- `quality: 'high' | 'medium' | 'low'` - Output quality
- `className?: string` - Additional CSS classes
- `layout?: 'horizontal' | 'vertical' | 'compact'` - Layout style (default: 'horizontal')
- `showIcons?: boolean` - Show performance icons (default: true)

### 4. StatusBadge

Model availability status with loading animations and retry functionality.

```tsx
<StatusBadge status="available" />
<StatusBadge status="error" showRetry onRetry={() => handleRetry()} />
<StatusBadge status="installing" variant="indicator" />
```

**Props:**
- `status: 'available' | 'installing' | 'error' | 'unavailable'` - Current status
- `className?: string` - Additional CSS classes
- `showRetry?: boolean` - Show retry button for error states (default: false)
- `onRetry?: () => void` - Retry callback function
- `size?: 'sm' | 'md' | 'lg'` - Size variant (default: 'md')
- `variant?: 'badge' | 'indicator'` - Display style (default: 'badge')

### 5. CapabilityBadge

Generic badge for various model capabilities with intelligent icon mapping.

```tsx
<CapabilityBadge capability="Vision" type="primary" />
<CapabilityBadge capability="Function Calls" type="success" />
<CapabilityBadge capability="Code Generation" showIcon={false} />
```

**Props:**
- `capability: string` - Capability name
- `className?: string` - Additional CSS classes
- `type?: 'primary' | 'secondary' | 'success' | 'warning'` - Color theme (default: 'secondary')
- `size?: 'sm' | 'md' | 'lg'` - Size variant (default: 'md')
- `showIcon?: boolean` - Show auto-detected icon (default: true)
- `description?: string` - Tooltip description

**Auto-detected Icons:**
- Vision/Image: Eye icon
- Function/Tool/API: Zap icon
- Code/Programming: Code icon
- Text/Language: FileText icon
- Embedding/Vector: Layers icon
- Chat/Conversation: MessageSquare icon
- Multimodal: Sparkles icon
- Reasoning/Intelligence: Brain icon

### 6. PricingIndicator

Cost per token display with multiple formats and relative cost indicators.

```tsx
<PricingIndicator pricing={{ input: 0.0001, output: 0.0002 }} />
<PricingIndicator showRelativeCost relativeCostLevel={3} />
<PricingIndicator pricing={{ input: 0.5, output: 1.5, unit: 'per_1k_tokens' }} layout="vertical" />
```

**Props:**
- `pricing?: { input?: number; output?: number; unit?: string }` - Detailed pricing info
- `className?: string` - Additional CSS classes
- `layout?: 'horizontal' | 'vertical' | 'compact'` - Layout style (default: 'horizontal')
- `showRelativeCost?: boolean` - Show relative cost level instead of detailed pricing
- `relativeCostLevel?: 1 | 2 | 3 | 4` - Relative cost level ($ to $$$$)
- `showIcon?: boolean` - Show dollar sign icon (default: true)

## Usage Examples

### In ModelSpecificationCard

```tsx
import {
  ToolCallIndicator,
  EmbeddingDimensionChip,
  PerformanceIndicator,
  StatusBadge,
  CapabilityBadge,
  PricingIndicator
} from '../ui/indicators';

export const ModelSpecificationCard = ({ model, isSelected, onSelect }) => {
  return (
    <div className="p-4 border rounded-lg">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold">{model.displayName}</h3>
        <StatusBadge status={model.status} size="sm" />
      </div>
      
      <div className="space-y-2">
        <div className="flex flex-wrap gap-2">
          {model.capabilities.map(cap => (
            <CapabilityBadge key={cap} capability={cap} size="sm" />
          ))}
          <ToolCallIndicator supported={model.toolSupport} showLabel={false} size="sm" />
          {model.dimensions && (
            <EmbeddingDimensionChip dimensions={model.dimensions} size="sm" />
          )}
        </div>
        
        <div className="flex items-center justify-between">
          <PerformanceIndicator 
            speed={model.performance.speed} 
            quality={model.performance.quality} 
            layout="compact" 
          />
          <PricingIndicator pricing={model.pricing} layout="compact" />
        </div>
      </div>
    </div>
  );
};
```

### In Provider Settings

```tsx
<div className="grid grid-cols-2 gap-4">
  {models.map(model => (
    <div key={model.id} className="p-3 border rounded">
      <div className="flex items-center gap-2 mb-2">
        <span className="font-medium">{model.name}</span>
        <StatusBadge 
          status={model.status} 
          variant="indicator" 
          size="sm"
          showRetry={model.status === 'error'}
          onRetry={() => retryModelInstall(model.id)}
        />
      </div>
      
      {model.type === 'embedding' && (
        <EmbeddingDimensionChip dimensions={model.dimensions} />
      )}
      
      <ToolCallIndicator 
        supported={model.toolSupport} 
        className="mt-2" 
      />
    </div>
  ))}
</div>
```

## Design Principles

1. **Consistent Visual Language**: All indicators follow the same design patterns and color schemes
2. **Contextual Information**: Hover tooltips provide additional details
3. **Flexible Sizing**: Multiple size variants for different use cases
4. **Responsive Design**: Components adapt to available space
5. **Accessibility**: Proper ARIA labels and keyboard navigation
6. **Performance**: Lightweight components with minimal re-renders

## Color Coding

- **Green**: Positive states (available, fast, high quality, affordable)
- **Blue**: Neutral/informational states (standard, balanced)
- **Orange**: Warning states (medium performance, premium pricing)
- **Red/Pink**: Error states (unavailable, failed)
- **Gray**: Inactive/unknown states
- **Purple**: Enhanced/premium features

## Integration Notes

These components are designed to work seamlessly with:
- Existing Badge component system
- TailwindCSS utility classes
- Lucide React icons
- Framer Motion animations (where applicable)
- Dark mode support