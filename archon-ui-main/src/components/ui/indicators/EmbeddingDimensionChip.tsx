import React from 'react';
import { Layers } from 'lucide-react';
import { Badge } from '../Badge';

interface EmbeddingDimensionChipProps {
  dimensions: number;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
}

export const EmbeddingDimensionChip: React.FC<EmbeddingDimensionChipProps> = ({
  dimensions,
  className = '',
  size = 'md',
  showIcon = true
}) => {
  // Color coding based on dimension size for performance characteristics
  const getDimensionConfig = (dims: number) => {
    if (dims <= 384) {
      return {
        color: 'green' as const,
        performance: 'Fast, lightweight embeddings',
        category: 'Compact'
      };
    } else if (dims <= 768) {
      return {
        color: 'blue' as const,
        performance: 'Balanced speed and quality',
        category: 'Standard'
      };
    } else if (dims <= 1536) {
      return {
        color: 'purple' as const,
        performance: 'High quality embeddings',
        category: 'Enhanced'
      };
    } else {
      return {
        color: 'orange' as const,
        performance: 'Maximum quality, slower processing',
        category: 'Premium'
      };
    }
  };

  const config = getDimensionConfig(dimensions);
  const sizeMap = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base'
  };

  const iconSizeMap = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5'
  };

  const title = `${dimensions}D embeddings - ${config.category}: ${config.performance}`;

  return (
    <div className={`flex items-center ${className}`} title={title}>
      <Badge 
        color={config.color} 
        variant="outline" 
        className={`${sizeMap[size]} flex items-center gap-1.5`}
      >
        {showIcon && <Layers className={iconSizeMap[size]} />}
        <span className="font-medium">{dimensions}D</span>
      </Badge>
    </div>
  );
};