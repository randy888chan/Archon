import React from 'react';
import { DollarSign, TrendingUp, TrendingDown } from 'lucide-react';
import { Badge } from '../Badge';

interface PricingIndicatorProps {
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

export const PricingIndicator: React.FC<PricingIndicatorProps> = ({
  pricing,
  className = '',
  layout = 'horizontal',
  showRelativeCost = false,
  relativeCostLevel,
  showIcon = true
}) => {
  // Format price with appropriate precision
  const formatPrice = (price: number, unit?: string) => {
    if (unit === 'per_1k_tokens' || price >= 0.001) {
      return price.toFixed(4);
    } else if (price >= 0.0001) {
      return price.toFixed(5);
    } else {
      return price.toExponential(2);
    }
  };

  // Get relative cost configuration
  const getRelativeCostConfig = (level: number) => {
    switch (level) {
      case 1:
        return {
          symbol: '$',
          color: 'green' as const,
          description: 'Very affordable'
        };
      case 2:
        return {
          symbol: '$$',
          color: 'blue' as const,
          description: 'Moderate cost'
        };
      case 3:
        return {
          symbol: '$$$',
          color: 'orange' as const,
          description: 'Premium pricing'
        };
      case 4:
        return {
          symbol: '$$$$',
          color: 'pink' as const,
          description: 'High-end pricing'
        };
      default:
        return {
          symbol: '?',
          color: 'gray' as const,
          description: 'Pricing unknown'
        };
    }
  };

  // If no pricing info available
  if (!pricing && !showRelativeCost) {
    return (
      <div className={`flex items-center gap-1 ${className}`}>
        <Badge color="gray" variant="outline" className="text-xs">
          Free
        </Badge>
      </div>
    );
  }

  // Show relative cost level
  if (showRelativeCost && relativeCostLevel) {
    const costConfig = getRelativeCostConfig(relativeCostLevel);
    return (
      <div className={`flex items-center gap-1.5 ${className}`} title={costConfig.description}>
        {showIcon && <DollarSign className="w-3 h-3 text-gray-500" />}
        <Badge color={costConfig.color} variant="outline" className="text-xs font-mono">
          {costConfig.symbol}
        </Badge>
      </div>
    );
  }

  // Show detailed pricing
  if (!pricing) return null;

  const layoutStyles = {
    horizontal: 'flex items-center gap-2',
    vertical: 'flex flex-col gap-1',
    compact: 'flex items-center gap-1'
  };

  const unitDisplay = pricing.unit === 'per_1k_tokens' ? '/1K' : '/token';

  if (layout === 'compact') {
    const avgPrice = pricing.input && pricing.output 
      ? (pricing.input + pricing.output) / 2 
      : pricing.input || pricing.output;
    
    return (
      <div 
        className={`${layoutStyles[layout]} ${className}`}
        title={`Average: $${avgPrice ? formatPrice(avgPrice)}${unitDisplay}`}
      >
        {showIcon && <DollarSign className="w-3 h-3 text-gray-500" />}
        <Badge color="blue" variant="outline" className="text-xs font-mono">
          ${avgPrice ? formatPrice(avgPrice) : '?'}{unitDisplay}
        </Badge>
      </div>
    );
  }

  return (
    <div className={`${layoutStyles[layout]} ${className}`}>
      {showIcon && layout !== 'vertical' && (
        <DollarSign className="w-3 h-3 text-gray-500" />
      )}
      
      <div className={layout === 'vertical' ? 'flex flex-col gap-1' : 'flex items-center gap-2'}>
        {pricing.input && (
          <div className="flex items-center gap-1" title={`Input cost: $${formatPrice(pricing.input)}${unitDisplay}`}>
            <TrendingDown className="w-3 h-3 text-green-500" />
            <Badge color="green" variant="outline" className="text-xs font-mono">
              ${formatPrice(pricing.input)}{unitDisplay}
            </Badge>
          </div>
        )}
        
        {pricing.output && (
          <div className="flex items-center gap-1" title={`Output cost: $${formatPrice(pricing.output)}${unitDisplay}`}>
            <TrendingUp className="w-3 h-3 text-orange-500" />
            <Badge color="orange" variant="outline" className="text-xs font-mono">
              ${formatPrice(pricing.output)}{unitDisplay}
            </Badge>
          </div>
        )}
      </div>
    </div>
  );
};