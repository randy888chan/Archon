import React from 'react';
import { Activity, Zap, Clock } from 'lucide-react';
import { Badge } from '../Badge';

interface PerformanceIndicatorProps {
  speed: 'fast' | 'medium' | 'slow';
  quality: 'high' | 'medium' | 'low';
  className?: string;
  layout?: 'horizontal' | 'vertical' | 'compact';
  showIcons?: boolean;
}

export const PerformanceIndicator: React.FC<PerformanceIndicatorProps> = ({
  speed,
  quality,
  className = '',
  layout = 'horizontal',
  showIcons = true
}) => {
  const getSpeedConfig = (speed: string) => {
    switch (speed) {
      case 'fast':
        return {
          color: 'green' as const,
          icon: <Zap className="w-3 h-3" />,
          label: 'Fast',
          description: 'Quick response times'
        };
      case 'medium':
        return {
          color: 'orange' as const,
          icon: <Activity className="w-3 h-3" />,
          label: 'Balanced',
          description: 'Moderate response times'
        };
      case 'slow':
        return {
          color: 'gray' as const,
          icon: <Clock className="w-3 h-3" />,
          label: 'Thorough',
          description: 'Slower but comprehensive'
        };
      default:
        return {
          color: 'gray' as const,
          icon: <Activity className="w-3 h-3" />,
          label: 'Unknown',
          description: 'Performance unknown'
        };
    }
  };

  const getQualityConfig = (quality: string) => {
    switch (quality) {
      case 'high':
        return {
          color: 'green' as const,
          label: 'High Quality',
          description: 'Superior output quality'
        };
      case 'medium':
        return {
          color: 'orange' as const,
          label: 'Good Quality',
          description: 'Reliable output quality'
        };
      case 'low':
        return {
          color: 'gray' as const,
          label: 'Basic Quality',
          description: 'Adequate output quality'
        };
      default:
        return {
          color: 'gray' as const,
          label: 'Unknown Quality',
          description: 'Quality unknown'
        };
    }
  };

  const speedConfig = getSpeedConfig(speed);
  const qualityConfig = getQualityConfig(quality);

  const layoutStyles = {
    horizontal: 'flex items-center gap-2',
    vertical: 'flex flex-col gap-1',
    compact: 'flex items-center gap-1'
  };

  if (layout === 'compact') {
    return (
      <div 
        className={`${layoutStyles[layout]} ${className}`}
        title={`Speed: ${speedConfig.description}, Quality: ${qualityConfig.description}`}
      >
        {showIcons && speedConfig.icon}
        <div className="flex gap-1">
          <Badge color={speedConfig.color} variant="outline" className="text-xs">
            {speedConfig.label}
          </Badge>
          <Badge color={qualityConfig.color} variant="outline" className="text-xs">
            {qualityConfig.label}
          </Badge>
        </div>
      </div>
    );
  }

  return (
    <div className={`${layoutStyles[layout]} ${className}`}>
      <div className="flex items-center gap-1.5" title={speedConfig.description}>
        {showIcons && speedConfig.icon}
        <Badge color={speedConfig.color} variant="outline" className="text-xs">
          {speedConfig.label}
        </Badge>
      </div>
      <div className="flex items-center gap-1.5" title={qualityConfig.description}>
        <Badge color={qualityConfig.color} variant="outline" className="text-xs">
          {qualityConfig.label}
        </Badge>
      </div>
    </div>
  );
};