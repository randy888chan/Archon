import React from 'react';
import { CheckCircle2, X } from 'lucide-react';
import { Badge } from '../Badge';

interface ToolCallIndicatorProps {
  supported: boolean;
  className?: string;
  showLabel?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export const ToolCallIndicator: React.FC<ToolCallIndicatorProps> = ({
  supported,
  className = '',
  showLabel = true,
  size = 'md'
}) => {
  const sizeMap = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5'
  };

  const config = supported
    ? {
        icon: <CheckCircle2 className={`${sizeMap[size]} text-green-500`} />,
        label: 'Function Calls',
        color: 'green' as const,
        title: 'This model supports function/tool calling capabilities'
      }
    : {
        icon: <X className={`${sizeMap[size]} text-gray-400`} />,
        label: 'No Function Calls',
        color: 'gray' as const,
        title: 'This model does not support function/tool calling'
      };

  if (!showLabel) {
    return (
      <div className={`flex items-center ${className}`} title={config.title}>
        {config.icon}
      </div>
    );
  }

  return (
    <div className={`flex items-center gap-1.5 ${className}`} title={config.title}>
      <Badge color={config.color} variant="outline" className="text-xs flex items-center gap-1">
        {config.icon}
        <span>{config.label}</span>
      </Badge>
    </div>
  );
};